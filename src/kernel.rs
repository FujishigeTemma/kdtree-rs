//! Two rules hold across every distance kernel here:
//!
//! - reduced distances accumulate monotonically, so a partial value already past
//!   the caller's bound is a valid answer;
//! - bound checks happen once per SIMD chunk: per axis the horizontal reduction
//!   serializes the kernel, per row the high-dimension early exit is gone.
//!
//! `powf` has no SIMD lowering, so a general `L^p` never reaches a vector kernel
//! — hence the unreachable `LP` arms of `axes_lanes` and `fold_scalar`.
//!
//! Narrowing the vector primitives to a three-variant enum removes those
//! `unreachable!`s but measured 10-25% slower on d8/d16 leaf scans with
//! `leafsize = 128`; cause unknown, so A/B it rather than assuming it is free.

use std::simd::prelude::*;
use std::simd::simd_swizzle;

use crate::layout::{BBox, BBoxMut, Rows, Width, axis_offset};
use crate::metric::{Metric, Rd};
use crate::simd::{F64s, LANES, hmax, hsum, nonfinite_lanes, vmax, vmin};

#[inline(always)]
fn axes_lanes(m: Metric, delta: F64s) -> F64s {
    match m {
        Metric::L1 | Metric::LInf => delta.abs(),
        Metric::L2 => delta * delta,
        Metric::LP(_) => unreachable!("LP never enters a vector kernel"),
    }
}

#[inline(always)]
fn combine<const N: usize>(m: Metric, a: Simd<f64, N>, b: Simd<f64, N>) -> Simd<f64, N> {
    match m {
        Metric::LInf => vmax(a, b),
        _ => a + b,
    }
}

/// The match stays outside the loop so LLVM can unroll or vectorize each arm.
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

pub(crate) fn all_finite(values: &[f64]) -> bool {
    let mut nonfinite = Mask::<i64, LANES>::splat(false);
    let (chunks, rest) = values.as_chunks::<LANES>();
    for c in chunks {
        nonfinite |= nonfinite_lanes(F64s::from_array(*c));
    }
    !nonfinite.any() && rest.iter().all(|v| v.is_finite())
}

/// Covers all of 1..=8, plus common wider even widths.
const MAX_PHASES: usize = 8;

impl<W: Width> Rows<'_, W> {
    pub(crate) fn bbox_into(self, out: &mut BBoxMut<'_>) {
        self.bbox_kernel::<false>(out);
    }

    /// `false` when any element is non-finite.
    pub(crate) fn bbox_checked_into(self, out: &mut BBoxMut<'_>) -> bool {
        self.bbox_kernel::<true>(out)
    }

    /// The block streams as flat `LANES`-wide vectors, so lane `j` of vector `i`
    /// holds dimension `(i * LANES + j) % ndim` — a pattern repeating every
    /// `ndim / gcd(ndim, LANES)` vectors, hence one accumulator per phase and a
    /// scalar merge to scatter their lanes back.
    fn bbox_kernel<const CHECK_FINITE: bool>(self, out: &mut BBoxMut<'_>) -> bool {
        let ndim = self.ndim();
        let lo = &mut out.lo[..ndim];
        let hi = &mut out.hi[..ndim];
        lo.fill(f64::INFINITY);
        hi.fill(f64::NEG_INFINITY);

        let data = self.flat();
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
            for coords in self.iter() {
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
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// `inline(always)`, not `inline`: with `codegen-units = 1` the size heuristic
/// depends on the whole crate, so growing anything anywhere can demote this to a
/// real call per point — 12% on d16 leaf scans when it happened. `nm | grep
/// point_rd` finding a symbol means it has.
#[inline(always)]
fn point_rd(m: Metric, lhs: &[f64], rhs: &[f64], bound: Rd) -> Rd {
    let bound = bound.get();
    Rd::reduced(match m {
        Metric::L1 | Metric::L2 => point_rd_sum(m, lhs, rhs, bound),
        Metric::LInf => point_rd_linf(lhs, rhs, bound),
        Metric::LP(p) => point_rd_lp(p, lhs, rhs, bound),
    })
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

/// `bound` is called once per scan plus once per accepted candidate, never per
/// point — [`Scan`] threads the value through its loops to keep it that way.
pub(crate) trait Sink {
    fn bound(&self) -> Rd;
    fn offer(&mut self, offset: usize, rd: Rd);
}

/// Resolved once per `query` call so each descent carries only the strategy it
/// uses. Inlining every strategy into one descent instead costs the short-row
/// queries the instruction cache for paths they never take (8-11% on d3), and
/// hoisting only [`Streamed`] out of line costs the wide-row queries the same.
pub(crate) trait Strategy {
    fn scan<S: Sink>(m: Metric, q: &[f64], block: Rows<'_>, sink: &mut S);
}

const MAX_PACKED_WIDTH: usize = LANES;

/// Two chunks is where the benchmark grid stops, so the threshold sits at the
/// edge of what is measured rather than at a guess about wider rows.
const MAX_FOLDED_WIDTH: usize = 2 * LANES;

pub(crate) enum Plan {
    Packed,
    FoldedAbs,
    FoldedSquared,
    Streamed,
}

pub(crate) fn plan(m: Metric, ndim: usize) -> Plan {
    match m {
        Metric::L1 | Metric::L2 | Metric::LInf if ndim <= MAX_PACKED_WIDTH => Plan::Packed,
        Metric::L1 if ndim <= MAX_FOLDED_WIDTH => Plan::FoldedAbs,
        Metric::L2 if ndim <= MAX_FOLDED_WIDTH => Plan::FoldedSquared,
        _ => Plan::Streamed,
    }
}

/// Rows short enough that [`point_rd`]'s early exit would save nothing: pack
/// several points per step and leave the SIMD domain only for the rare in-bound
/// candidates.
pub(crate) struct Packed;

impl Strategy for Packed {
    fn scan<S: Sink>(m: Metric, q: &[f64], block: Rows<'_>, sink: &mut S) {
        let bound = sink.bound();
        let mut scan = Scan {
            m,
            q,
            flat: block.flat(),
            sink,
        };
        match block.ndim() {
            1 => scan.unrolled::<1>(bound),
            2 => scan.d2(bound),
            3 => scan.d3(bound),
            4 => scan.d4(bound),
            5 => scan.unrolled::<5>(bound),
            6 => scan.unrolled::<6>(bound),
            7 => scan.unrolled::<7>(bound),
            8 => scan.d8(bound),
            _ => unreachable!("`packs` routes wider rows to `Streamed`"),
        };
    }
}

/// Rows long enough that the early exit beats packing — and the only option for
/// `L^p`.
pub(crate) struct Streamed;

impl Strategy for Streamed {
    fn scan<S: Sink>(m: Metric, q: &[f64], block: Rows<'_>, sink: &mut S) {
        let bound = sink.bound();
        Scan {
            m,
            q,
            flat: block.flat(),
            sink,
        }
        .streamed(bound);
    }
}

/// Rows too wide for [`Packed`] but too short for [`Streamed`]'s mid-row bound
/// check to pay for itself: the whole row accumulates lane-wise and reduces once.
///
/// Measured on d16, which is two chunks: -38% / -57% for `L1` and -12% / -12%
/// for `L2`, at `leafsize` 8 / 128 on aarch64. `L2` gains less because its
/// squared axes cross the bound more often, which is what pays for the check —
/// extend [`MAX_FOLDED_WIDTH`] and that balance tips back.
///
/// Every loop that reaches a leaf has to stay monomorphic, hence the const axis
/// reduction and a separate strategy: adding an `L1` arm to [`point_rd`] cost the
/// *`L2`* d16 scans 12-14% on aarch64, reproducibly across three spellings, and
/// padding the baseline with unrelated code moved it only 3%.
pub(crate) struct Folded<const SQUARED: bool>;

impl<const SQUARED: bool> Strategy for Folded<SQUARED> {
    fn scan<S: Sink>(m: Metric, q: &[f64], block: Rows<'_>, sink: &mut S) {
        debug_assert!(match m {
            Metric::L1 => !SQUARED,
            Metric::L2 => SQUARED,
            _ => false,
        });
        let bound = sink.bound();
        Scan {
            m,
            q,
            flat: block.flat(),
            sink,
        }
        .folded::<SQUARED>(bound);
    }
}

/// `SQUARED` picks the axis reduction at compile time, so the `Metric` handed to
/// [`axes_lanes`] and [`fold_scalar`] is a constant and both matches vanish.
#[inline(always)]
fn point_rd_folded<const SQUARED: bool>(lhs: &[f64], rhs: &[f64]) -> f64 {
    let m = if SQUARED { Metric::L2 } else { Metric::L1 };
    let (l_chunks, l_rest) = lhs.as_chunks::<LANES>();
    let (r_chunks, r_rest) = rhs.as_chunks::<LANES>();
    let mut acc = F64s::splat(0.0);
    for (l, r) in l_chunks.iter().zip(r_chunks) {
        acc += axes_lanes(m, F64s::from_array(*l) - F64s::from_array(*r));
    }
    let rd = if l_chunks.is_empty() { 0.0 } else { hsum(acc) };
    fold_scalar(m, rd, l_rest, r_rest)
}

/// The pruning bound is deliberately not a field: it flows through every method
/// as a value to stay in a register. As a field behind `&mut self` it becomes a
/// load per point, ~10% on the short-row kernels.
struct Scan<'a, S: Sink> {
    m: Metric,
    q: &'a [f64],
    flat: &'a [f64],
    sink: &'a mut S,
}

impl<S: Sink> Scan<'_, S> {
    #[inline(always)]
    fn offer(&mut self, offset: usize, rd: Rd) -> Rd {
        self.sink.offer(offset, rd);
        self.sink.bound()
    }

    #[inline(always)]
    fn folded<const SQUARED: bool>(&mut self, mut bound: Rd) -> Rd {
        let (q, flat) = (self.q, self.flat);
        for (j, coords) in flat.chunks_exact(q.len()).enumerate() {
            let rd = Rd::reduced(point_rd_folded::<SQUARED>(q, coords));
            if rd <= bound {
                bound = self.offer(j, rd);
            }
        }
        bound
    }

    #[inline(always)]
    fn streamed(&mut self, mut bound: Rd) -> Rd {
        let (m, q, flat) = (self.m, self.q, self.flat);
        for (j, coords) in flat.chunks_exact(q.len()).enumerate() {
            let rd = point_rd(m, q, coords, bound);
            if rd <= bound {
                bound = self.offer(j, rd);
            }
        }
        bound
    }

    /// A compile-time width unrolls the axis loop, and with no early exit LLVM
    /// vectorizes across points.
    #[inline(always)]
    fn unrolled<const D: usize>(&mut self, bound: Rd) -> Rd {
        let flat = self.flat;
        self.unrolled_from::<D>(flat, 0, bound)
    }

    #[inline(always)]
    fn unrolled_from<const D: usize>(&mut self, rows: &[f64], base: usize, mut bound: Rd) -> Rd {
        let (m, q) = (self.m, self.q);
        let q = q.first_chunk::<D>().expect("row width exceeds query");
        for (j, p) in rows.as_chunks::<D>().0.iter().enumerate() {
            let rd = Rd::reduced(fold_scalar(m, 0.0, q, p));
            if rd <= bound {
                bound = self.offer(base + j, rd);
            }
        }
        bound
    }

    /// Walk the block in `CHUNK`-element groups of `P` points, turn each into a
    /// `P`-lane reduced-distance vector, and fan the rare in-bound lanes out to
    /// the sink. Every kernel below is this driver plus a swizzle recipe.
    #[inline(always)]
    fn packed<const CHUNK: usize, const P: usize, const D: usize>(
        &mut self,
        mut bound: Rd,
        rds_of: impl Fn(&[f64; CHUNK]) -> Simd<f64, P>,
    ) -> Rd {
        debug_assert_eq!(CHUNK, P * D);
        let (chunks, rest) = self.flat.as_chunks::<CHUNK>();
        let mut bound_v = Simd::splat(bound.get());
        for (i, c) in chunks.iter().enumerate() {
            let rds = rds_of(c);
            if rds.simd_le(bound_v).any() {
                bound = self.emit(rds, i * P, bound);
                bound_v = Simd::splat(bound.get());
            }
        }
        self.unrolled_from::<D>(rest, chunks.len() * P, bound)
    }

    #[inline(always)]
    fn emit<const N: usize>(&mut self, rds: Simd<f64, N>, base: usize, mut bound: Rd) -> Rd {
        for (j, &rd) in rds.as_array().iter().enumerate() {
            let rd = Rd::reduced(rd);
            if rd <= bound {
                bound = self.offer(base + j, rd);
            }
        }
        bound
    }

    /// Four 2-d points per register: `[x0 y0 x1 y1 x2 y2 x3 y3]`.
    #[inline(always)]
    fn d2(&mut self, bound: Rd) -> Rd {
        let (m, q) = (self.m, self.q);
        let pattern = F64s::from_array([q[0], q[1], q[0], q[1], q[0], q[1], q[0], q[1]]);
        self.packed::<LANES, 4, 2>(bound, |c| {
            let axes = axes_lanes(m, F64s::from_array(*c) - pattern);
            let x = simd_swizzle!(axes, [0, 2, 4, 6]);
            let y = simd_swizzle!(axes, [1, 3, 5, 7]);
            combine(m, x, y)
        })
    }

    /// Eight 3-d points across three registers: 24 lanes hold
    /// `[p0.xyz p1.xyz .. p7.xyz]`. An odd width aligns to no register boundary,
    /// so each axis vector is gathered from all three sources.
    #[inline(always)]
    fn d3(&mut self, bound: Rd) -> Rd {
        let (m, q) = (self.m, self.q);
        let pat0 = F64s::from_array([q[0], q[1], q[2], q[0], q[1], q[2], q[0], q[1]]);
        let pat1 = F64s::from_array([q[2], q[0], q[1], q[2], q[0], q[1], q[2], q[0]]);
        let pat2 = F64s::from_array([q[1], q[2], q[0], q[1], q[2], q[0], q[1], q[2]]);
        self.packed::<{ 3 * LANES }, LANES, 3>(bound, |c| {
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
        })
    }

    /// Two 4-d points per register: `[p0: 0..4 | p1: 0..4]`.
    #[inline(always)]
    fn d4(&mut self, bound: Rd) -> Rd {
        let (m, q) = (self.m, self.q);
        let pattern = F64s::from_array([q[0], q[1], q[2], q[3], q[0], q[1], q[2], q[3]]);
        self.packed::<LANES, 2, 4>(bound, |c| {
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
        })
    }

    /// At `d == LANES` a point already fills a register: eight whole vectors
    /// folded together rather than one register de-interleaved. Reducing each
    /// point on its own would cost an `hsum` apiece and serialize on one
    /// dependency chain, which at this width the early exit cannot leave.
    #[inline(always)]
    fn d8(&mut self, bound: Rd) -> Rd {
        let (m, q) = (self.m, self.q);
        let pattern = F64s::from_slice(q);
        self.packed::<{ LANES * LANES }, LANES, LANES>(bound, |c| {
            let axes: [F64s; LANES] =
                std::array::from_fn(|i| axes_lanes(m, F64s::from_slice(&c[i * LANES..]) - pattern));
            // `fold(a, b)` pairs adjacent lanes within each source, leaving a's
            // partials in lanes 0..4 and b's in 4..8; three rounds land the fold
            // of `axes[j]` in lane `j`.
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
        })
    }
}

/// Zero when `q` is inside `b`. Comparable against [`point_rd`].
#[inline]
pub(crate) fn box_rd(m: Metric, q: &[f64], b: BBox<'_>) -> Rd {
    Rd::reduced(match m {
        Metric::L1 | Metric::L2 => box_rd_sum(m, q, b.lo, b.hi),
        Metric::LInf => box_rd_linf(q, b.lo, b.hi),
        Metric::LP(p) => box_rd_lp(p, q, b.lo, b.hi),
    })
}

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
        rd += m.reduce(axis_offset(qs, ls, hs)).get();
    }
    rd
}

fn box_rd_linf(q: &[f64], lo: &[f64], hi: &[f64]) -> f64 {
    let mut worst = 0.0_f64;
    for ((&qs, &ls), &hs) in q.iter().zip(lo).zip(hi) {
        let off = axis_offset(qs, ls, hs);
        if off > worst {
            worst = off;
        }
    }
    worst
}

fn box_rd_lp(p: f64, q: &[f64], lo: &[f64], hi: &[f64]) -> f64 {
    let mut rd = 0.0_f64;
    for ((&qs, &ls), &hs) in q.iter().zip(lo).zip(hi) {
        let off = axis_offset(qs, ls, hs);
        if off > 0.0 {
            rd += off.powf(p);
        }
    }
    rd
}
