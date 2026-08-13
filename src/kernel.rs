//! Vectorized bulk kernels: every loop that streams whole blocks of point
//! rows lives here. The build uses [`bbox`]/[`bbox_check_finite`] to bound
//! row blocks; queries use the reduced-distance kernels [`point_rd`],
//! [`scan_leaf`], and [`box_rd`] (all values in the domain of `metric.rs`).
//!
//! The distance kernels share two ideas:
//! - reduced distances accumulate monotonically for every supported metric,
//!   so a partial value that already exceeds the caller's bound is a valid
//!   early-exit result;
//! - bound checks happen once per SIMD chunk, not per axis — a horizontal
//!   reduction every axis would serialize the kernel, while a full-row scan
//!   would give up the early exit that saves most of the work in high
//!   dimensions.
//!
//! # `L^p` routing
//!
//! `powf` has no SIMD lowering, so a general `L^p` never enters a vector
//! kernel: each public entry point (`point_rd`, `scan_leaf`, `box_rd`)
//! diverts `Metric::LP` to its own scalar path up front, and the three
//! vector primitives below (`axes_lanes`, `combine`, `fold_scalar`) are
//! unreachable for it — hence their `unreachable!` arms.
//!
//! Narrowing those primitives to a three-variant enum removes the
//! `unreachable!`s and reads better, but measured 10-25% slower on
//! high-dimensional leaf scans (d8/d16 with `leafsize = 128`), so `Metric`
//! stays the parameter type. Only the effect was measured, not the cause;
//! if you retry this, A/B it rather than trusting that it is free.

use std::simd::prelude::*;
use std::simd::simd_swizzle;

use crate::metric::{Metric, box_axis_offset};
use crate::simd::{F64s, LANES, hmax, hsum, nonfinite_lanes, vmax, vmin};

/// Reduced contribution of a lane-wise axis difference.
#[inline(always)]
fn axes_lanes(m: Metric, delta: F64s) -> F64s {
    match m {
        Metric::L1 | Metric::LInf => delta.abs(),
        Metric::L2 => delta * delta,
        Metric::LP(_) => unreachable!("LP never enters a vector kernel"),
    }
}

/// Combine two vectors of partial reduced distances lane-wise: max for
/// `L^inf`, sum for every other metric.
#[inline(always)]
fn combine<const N: usize>(m: Metric, a: Simd<f64, N>, b: Simd<f64, N>) -> Simd<f64, N> {
    match m {
        Metric::LInf => vmax(a, b),
        _ => a + b,
    }
}

/// Scalar fold over any run of axes. The match stays outside the loop so
/// each arm keeps a tight body LLVM can unroll or vectorize.
#[inline(always)]
fn fold_scalar(m: Metric, mut rd: f64, lhs: &[f64], rhs: &[f64]) -> f64 {
    match m {
        Metric::L1 => {
            for (a, b) in lhs.iter().zip(rhs) {
                rd += (a - b).abs();
            }
        }
        Metric::L2 => {
            for (a, b) in lhs.iter().zip(rhs) {
                let delta = a - b;
                rd += delta * delta;
            }
        }
        Metric::LInf => {
            for (a, b) in lhs.iter().zip(rhs) {
                let delta = (a - b).abs();
                if delta > rd {
                    rd = delta;
                }
            }
        }
        Metric::LP(_) => unreachable!("LP folds only in point_rd_lp"),
    }
    rd
}

// --- finiteness --------------------------------------------------------------

/// Vectorized finiteness sweep over a whole slice. The build gets the same
/// check for free inside [`bbox_check_finite`]; queries, which compute no
/// box, use this standalone pass.
pub(crate) fn all_finite(values: &[f64]) -> bool {
    let mut nonfinite = Mask::<i64, LANES>::splat(false);
    let (chunks, rest) = values.as_chunks::<LANES>();
    for c in chunks {
        nonfinite |= nonfinite_lanes(F64s::from_array(*c));
    }
    !nonfinite.any() && rest.iter().all(|v| v.is_finite())
}

// --- bounding boxes of row blocks ------------------------------------------

/// Maximum number of phase accumulators for the flat bbox kernel: covers
/// every row width whose pattern period `ndim / gcd(ndim, LANES)` is at most
/// this many vectors (all of 1..=8, plus common wider even widths).
const MAX_PHASES: usize = 8;

/// Tight bounding box of the row-major block `data`, written to `lo`/`hi`
/// (each `ndim` long).
///
/// Instead of chunking each row (useless for rows shorter than a vector),
/// the kernel streams the block as flat `LANES`-wide vectors. Lane `j` of
/// flat vector `i` holds dimension `(i * LANES + j) % ndim`, a pattern that
/// repeats every `ndim / gcd(ndim, LANES)` vectors, so that many phase
/// accumulators cover every dimension; a final scalar merge scatters the
/// accumulator lanes back to their dimensions. Row widths with too long a
/// period fall back to a scalar row sweep.
///
/// `D` is the compile-time row width (`0` = dynamic): with it fixed, the
/// phase count is a constant and the whole phase/merge machinery unrolls.
pub(crate) fn bbox<const D: usize>(data: &[f64], ndim: usize, lo: &mut [f64], hi: &mut [f64]) {
    bbox_kernel::<D, false>(data, ndim, lo, hi);
}

/// [`bbox`], plus a fused finiteness check over every element (the
/// `v * 0 != 0` trick catches both infinities and NaNs without extra
/// passes). Returns `false` when any element is non-finite.
pub(crate) fn bbox_check_finite(data: &[f64], ndim: usize, lo: &mut [f64], hi: &mut [f64]) -> bool {
    bbox_kernel::<0, true>(data, ndim, lo, hi)
}

fn bbox_kernel<const D: usize, const CHECK_FINITE: bool>(
    data: &[f64],
    ndim: usize,
    lo: &mut [f64],
    hi: &mut [f64],
) -> bool {
    debug_assert!(D == 0 || D == ndim);
    let ndim = if D > 0 { D } else { ndim };
    let lo = &mut lo[..ndim];
    let hi = &mut hi[..ndim];
    lo.fill(f64::INFINITY);
    hi.fill(f64::NEG_INFINITY);

    let phases = ndim / gcd(ndim, LANES);
    let mut nonfinite = Mask::<i64, LANES>::splat(false);
    if phases <= MAX_PHASES {
        let mut acc_lo = [F64s::splat(f64::INFINITY); MAX_PHASES];
        let mut acc_hi = [F64s::splat(f64::NEG_INFINITY); MAX_PHASES];
        let (chunks, rest) = data.as_chunks::<LANES>();
        let mut phase = 0;
        for c in chunks {
            let v = F64s::from_array(*c);
            acc_lo[phase] = vmin(v, acc_lo[phase]);
            acc_hi[phase] = vmax(v, acc_hi[phase]);
            if CHECK_FINITE {
                nonfinite |= nonfinite_lanes(v);
            }
            phase += 1;
            if phase == phases {
                phase = 0;
            }
        }
        for (i, (al, ah)) in acc_lo[..phases].iter().zip(&acc_hi[..phases]).enumerate() {
            for j in 0..LANES {
                let dim = (i * LANES + j) % ndim;
                lo[dim] = lo[dim].min(al[j]);
                hi[dim] = hi[dim].max(ah[j]);
            }
        }
        let mut dim = (chunks.len() * LANES) % ndim;
        for &v in rest {
            lo[dim] = lo[dim].min(v);
            hi[dim] = hi[dim].max(v);
            if CHECK_FINITE && !v.is_finite() {
                return false;
            }
            dim += 1;
            if dim == ndim {
                dim = 0;
            }
        }
    } else {
        for coords in data.chunks_exact(ndim) {
            for ((l, h), &v) in lo.iter_mut().zip(hi.iter_mut()).zip(coords) {
                *l = l.min(v);
                *h = h.max(v);
                if CHECK_FINITE && !v.is_finite() {
                    return false;
                }
            }
        }
    }
    !CHECK_FINITE || !nonfinite.any()
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

// --- query to point ---------------------------------------------------------

/// Reduced distance between two points, returning early as soon as a chunk's
/// running total exceeds `bound`.
#[inline]
pub(crate) fn point_rd(m: Metric, lhs: &[f64], rhs: &[f64], bound: f64) -> f64 {
    match m {
        Metric::L1 | Metric::L2 => point_rd_sum(m, lhs, rhs, bound),
        Metric::LInf => point_rd_linf(lhs, rhs, bound),
        Metric::LP(p) => point_rd_lp(p, lhs, rhs, bound),
    }
}

fn point_rd_sum(m: Metric, lhs: &[f64], rhs: &[f64], bound: f64) -> f64 {
    let (l_chunks, l_rest) = lhs.as_chunks::<LANES>();
    let (r_chunks, r_rest) = rhs.as_chunks::<LANES>();
    let mut rd = 0.0_f64;
    for (l, r) in l_chunks.iter().zip(r_chunks) {
        rd += hsum(axes_lanes(m, F64s::from_array(*l) - F64s::from_array(*r)));
        if rd > bound {
            return rd;
        }
    }
    fold_scalar(m, rd, l_rest, r_rest)
}

/// `L^inf` needs neither a horizontal reduction per chunk nor a scalar
/// accumulator: the running maximum stays lane-wise, the early exit is a
/// mask test, and a single horizontal max happens on the way out.
fn point_rd_linf(lhs: &[f64], rhs: &[f64], bound: f64) -> f64 {
    let (l_chunks, l_rest) = lhs.as_chunks::<LANES>();
    let (r_chunks, r_rest) = rhs.as_chunks::<LANES>();
    if l_chunks.is_empty() {
        return fold_scalar(Metric::LInf, 0.0, l_rest, r_rest);
    }
    let mut vm = F64s::splat(0.0);
    let bound_v = F64s::splat(bound);
    for (l, r) in l_chunks.iter().zip(r_chunks) {
        vm = vmax(vm, (F64s::from_array(*l) - F64s::from_array(*r)).abs());
        if vm.simd_gt(bound_v).any() {
            return hmax(vm);
        }
    }
    fold_scalar(Metric::LInf, hmax(vm), l_rest, r_rest)
}

/// `powf` has no SIMD lowering, so `L^p` keeps a scalar loop with the same
/// chunk-granular early exit as the SIMD paths.
fn point_rd_lp(p: f64, lhs: &[f64], rhs: &[f64], bound: f64) -> f64 {
    let mut rd = 0.0_f64;
    for (l, r) in lhs.chunks(LANES).zip(rhs.chunks(LANES)) {
        for (a, b) in l.iter().zip(r) {
            rd += (a - b).abs().powf(p);
        }
        if rd > bound {
            return rd;
        }
    }
    rd
}

// --- query to leaf block ----------------------------------------------------

/// Scan a whole leaf block against `bound`, invoking `on_hit(offset,
/// reduced_distance)` for every point within it. `on_hit` returns the
/// tightened bound to use from that point on.
///
/// Rows of 1-4 dims are too short for [`point_rd`]'s early exit to save
/// anything, so they get branch-free kernels that vectorize across points
/// and keep distances in registers, leaving the SIMD domain only for the
/// (rare) in-bound candidates: d1 rows are contiguous scalars LLVM
/// vectorizes directly, d2-d4 get one swizzle recipe each on the shared
/// [`scan_chunks`] driver. Everything else scans point by point with the
/// early exit.
pub(crate) fn scan_leaf(
    m: Metric,
    q: &[f64],
    block: &[f64],
    mut bound: f64,
    mut on_hit: impl FnMut(usize, f64) -> f64,
) {
    if let Metric::LP(_) = m {
        return scan_points(m, q, block, bound, &mut on_hit);
    }
    match q.len() {
        1 => scan_rows::<1>(m, q, block, 0, &mut bound, &mut on_hit),
        2 => scan_d2(m, q, block, &mut bound, &mut on_hit),
        3 => scan_d3(m, q, block, &mut bound, &mut on_hit),
        4 => scan_d4(m, q, block, &mut bound, &mut on_hit),
        5 => scan_rows::<5>(m, q, block, 0, &mut bound, &mut on_hit),
        6 => scan_rows::<6>(m, q, block, 0, &mut bound, &mut on_hit),
        7 => scan_rows::<7>(m, q, block, 0, &mut bound, &mut on_hit),
        8 => scan_d8(m, q, block, &mut bound, &mut on_hit),
        _ => scan_points(m, q, block, bound, &mut on_hit),
    }
}

/// Per-point scan whose [`point_rd`] early exit skips most of each row once
/// the bound tightens.
fn scan_points(
    m: Metric,
    q: &[f64],
    block: &[f64],
    mut bound: f64,
    on_hit: &mut impl FnMut(usize, f64) -> f64,
) {
    let ndim = q.len();
    for (j, coords) in block.chunks_exact(ndim).enumerate() {
        let rd = point_rd(m, q, coords, bound);
        if rd <= bound {
            bound = on_hit(j, rd);
        }
    }
}

/// Scalar per-point scan for a compile-time row width: the axis loop fully
/// unrolls, and with no early exit LLVM vectorizes it across points. Also
/// serves as the sub-chunk tail of the d2-d4 kernels.
#[inline(always)]
fn scan_rows<const D: usize>(
    m: Metric,
    q: &[f64],
    rows: &[f64],
    base: usize,
    bound: &mut f64,
    on_hit: &mut impl FnMut(usize, f64) -> f64,
) {
    let q = q.first_chunk::<D>().expect("row width exceeds query");
    for (j, p) in rows.as_chunks::<D>().0.iter().enumerate() {
        let rd = fold_scalar(m, 0.0, q, p);
        if rd <= *bound {
            *bound = on_hit(base + j, rd);
        }
    }
}

/// One 8-d point per register, eight points per chunk.
///
/// Every other block kernel packs several short points into one register and
/// de-interleaves them; at `d == LANES` a point already fills a register, so
/// the shape is the mirror image — eight independent vectors folded together.
/// `scan_points` would instead reduce each point on its own, which costs a
/// full `hsum` per point and, worse, serializes on one dependency chain that
/// the `rd > bound` test cannot even exit early from at this width (a
/// `LANES`-wide row is a single chunk). Three butterfly rounds fold all eight
/// at once: half the shuffles, eight independent chains.
fn scan_d8(
    m: Metric,
    q: &[f64],
    block: &[f64],
    bound: &mut f64,
    on_hit: &mut impl FnMut(usize, f64) -> f64,
) {
    let pattern = F64s::from_slice(q);
    scan_chunks::<{ LANES * LANES }, LANES, LANES>(m, q, block, bound, on_hit, |c| {
        let axes: [F64s; LANES] =
            std::array::from_fn(|i| axes_lanes(m, F64s::from_slice(&c[i * LANES..]) - pattern));
        // `fold(a, b)` pairs adjacent lanes within each source, leaving a's
        // partials in lanes 0..4 and b's in lanes 4..8; three rounds thus
        // land the fold of `axes[j]` in lane `j`.
        let fold = |a: F64s, b: F64s| {
            combine(
                m,
                simd_swizzle!(a, b, [0, 2, 4, 6, 8, 10, 12, 14]),
                simd_swizzle!(a, b, [1, 3, 5, 7, 9, 11, 13, 15]),
            )
        };
        let r0 = fold(axes[0], axes[1]);
        let r1 = fold(axes[2], axes[3]);
        let r2 = fold(axes[4], axes[5]);
        let r3 = fold(axes[6], axes[7]);
        fold(fold(r0, r1), fold(r2, r3))
    });
}

/// Drive one branch-free block kernel: walk `block` in `CHUNK`-element
/// groups of `P` points, let `rds_of` turn each group into a `P`-lane
/// reduced-distance vector, and fan the (rare) in-bound lanes out to
/// `on_hit`. The sub-chunk leftover falls through to the scalar `D`-wide row
/// scan. The d2-d4 kernels are this driver plus a swizzle recipe.
#[inline(always)]
fn scan_chunks<const CHUNK: usize, const P: usize, const D: usize>(
    m: Metric,
    q: &[f64],
    block: &[f64],
    bound: &mut f64,
    on_hit: &mut impl FnMut(usize, f64) -> f64,
    rds_of: impl Fn(&[f64; CHUNK]) -> Simd<f64, P>,
) {
    debug_assert_eq!(CHUNK, P * D);
    let (chunks, rest) = block.as_chunks::<CHUNK>();
    let mut bound_v = Simd::splat(*bound);
    for (i, c) in chunks.iter().enumerate() {
        let rds = rds_of(c);
        if rds.simd_le(bound_v).any() {
            emit_hits(rds, i * P, bound, on_hit);
            bound_v = Simd::splat(*bound);
        }
    }
    scan_rows::<D>(m, q, rest, chunks.len() * P, bound, on_hit);
}

/// Four 2-d points per register: lanes hold `[x0 y0 x1 y1 x2 y2 x3 y3]`.
fn scan_d2(
    m: Metric,
    q: &[f64],
    block: &[f64],
    bound: &mut f64,
    on_hit: &mut impl FnMut(usize, f64) -> f64,
) {
    let pattern = F64s::from_array([q[0], q[1], q[0], q[1], q[0], q[1], q[0], q[1]]);
    scan_chunks::<LANES, 4, 2>(m, q, block, bound, on_hit, |c| {
        let axes = axes_lanes(m, F64s::from_array(*c) - pattern);
        let x = simd_swizzle!(axes, [0, 2, 4, 6]);
        let y = simd_swizzle!(axes, [1, 3, 5, 7]);
        combine(m, x, y)
    });
}

/// Eight 3-d points per three registers: 24 contiguous lanes hold
/// `[p0.xyz p1.xyz .. p7.xyz]`. An odd row width never aligns points to
/// register boundaries, so two-vector swizzles de-interleave the three
/// axis vectors into per-axis vectors (each needs two swizzles because its
/// lanes span all three sources) before the lane-wise combine.
fn scan_d3(
    m: Metric,
    q: &[f64],
    block: &[f64],
    bound: &mut f64,
    on_hit: &mut impl FnMut(usize, f64) -> f64,
) {
    let pat0 = F64s::from_array([q[0], q[1], q[2], q[0], q[1], q[2], q[0], q[1]]);
    let pat1 = F64s::from_array([q[2], q[0], q[1], q[2], q[0], q[1], q[2], q[0]]);
    let pat2 = F64s::from_array([q[1], q[2], q[0], q[1], q[2], q[0], q[1], q[2]]);
    scan_chunks::<{ 3 * LANES }, LANES, 3>(m, q, block, bound, on_hit, |c| {
        let a0 = axes_lanes(m, F64s::from_slice(&c[..8]) - pat0);
        let a1 = axes_lanes(m, F64s::from_slice(&c[8..16]) - pat1);
        let a2 = axes_lanes(m, F64s::from_slice(&c[16..]) - pat2);
        let xs = simd_swizzle!(
            simd_swizzle!(a0, a1, [0, 3, 6, 9, 12, 15, 0, 0]),
            a2,
            [0, 1, 2, 3, 4, 5, 10, 13]
        );
        let ys = simd_swizzle!(
            simd_swizzle!(a0, a1, [1, 4, 7, 10, 13, 0, 0, 0]),
            a2,
            [0, 1, 2, 3, 4, 8, 11, 14]
        );
        let zs = simd_swizzle!(
            simd_swizzle!(a0, a1, [2, 5, 8, 11, 14, 0, 0, 0]),
            a2,
            [0, 1, 2, 3, 4, 9, 12, 15]
        );
        combine(m, combine(m, xs, ys), zs)
    });
}

/// Two 4-d points per register: lanes hold `[p0: 0..4 | p1: 0..4]`.
fn scan_d4(
    m: Metric,
    q: &[f64],
    block: &[f64],
    bound: &mut f64,
    on_hit: &mut impl FnMut(usize, f64) -> f64,
) {
    let pattern = F64s::from_array([q[0], q[1], q[2], q[3], q[0], q[1], q[2], q[3]]);
    scan_chunks::<LANES, 2, 4>(m, q, block, bound, on_hit, |c| {
        let axes = axes_lanes(m, F64s::from_array(*c) - pattern);
        let pairs = combine(
            m,
            simd_swizzle!(axes, [0, 1, 4, 5]),
            simd_swizzle!(axes, [2, 3, 6, 7]),
        );
        combine(
            m,
            simd_swizzle!(pairs, [0, 2]),
            simd_swizzle!(pairs, [1, 3]),
        )
    });
}

/// Fan the in-bound lanes of a reduced-distance vector out to `on_hit`.
/// Callers re-splat their vector bound afterwards, since a hit tightens it.
#[inline(always)]
fn emit_hits<const N: usize>(
    rds: Simd<f64, N>,
    base: usize,
    bound: &mut f64,
    on_hit: &mut impl FnMut(usize, f64) -> f64,
) {
    for (j, &rd) in rds.as_array().iter().enumerate() {
        if rd <= *bound {
            *bound = on_hit(base + j, rd);
        }
    }
}

// --- query to bounding box ---------------------------------------------------

/// Reduced distance from `q` to the axis-aligned box `[lo, hi]`: zero when
/// `q` is inside. Comparable against the same reduced domain as
/// [`point_rd`].
#[inline]
pub(crate) fn box_rd(m: Metric, q: &[f64], lo: &[f64], hi: &[f64]) -> f64 {
    match m {
        Metric::L1 | Metric::L2 => box_rd_sum(m, q, lo, hi),
        Metric::LInf => box_rd_linf(q, lo, hi),
        Metric::LP(p) => box_rd_lp(p, q, lo, hi),
    }
}

/// Summing metrics (`L1`/`L2`) only; `L^inf` has its own max fold below.
fn box_rd_sum(m: Metric, q: &[f64], lo: &[f64], hi: &[f64]) -> f64 {
    let (q_chunks, q_rest) = q.as_chunks::<LANES>();
    let (lo_chunks, lo_rest) = lo.as_chunks::<LANES>();
    let (hi_chunks, hi_rest) = hi.as_chunks::<LANES>();
    let zero = F64s::splat(0.0);
    let mut acc = zero;
    for ((qc, lc), hc) in q_chunks.iter().zip(lo_chunks).zip(hi_chunks) {
        let qv = F64s::from_array(*qc);
        let off = vmax(
            vmax(F64s::from_array(*lc) - qv, qv - F64s::from_array(*hc)),
            zero,
        );
        acc += axes_lanes(m, off);
    }
    let mut rd = if q_chunks.is_empty() { 0.0 } else { hsum(acc) };
    for ((&qs, &ls), &hs) in q_rest.iter().zip(lo_rest).zip(hi_rest) {
        rd += m.axis_rd(box_axis_offset(qs, ls, hs));
    }
    rd
}

fn box_rd_linf(q: &[f64], lo: &[f64], hi: &[f64]) -> f64 {
    let mut worst = 0.0_f64;
    for ((&qs, &ls), &hs) in q.iter().zip(lo).zip(hi) {
        let off = box_axis_offset(qs, ls, hs);
        if off > worst {
            worst = off;
        }
    }
    worst
}

fn box_rd_lp(p: f64, q: &[f64], lo: &[f64], hi: &[f64]) -> f64 {
    let mut rd = 0.0_f64;
    for ((&qs, &ls), &hs) in q.iter().zip(lo).zip(hi) {
        let off = box_axis_offset(qs, ls, hs);
        if off > 0.0 {
            rd += off.powf(p);
        }
    }
    rd
}
