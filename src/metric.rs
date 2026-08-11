use std::simd::prelude::*;
use std::simd::simd_swizzle;

use crate::error::KDTreeError;
use crate::simd::{F64s, LANES, hmax, hsum, vmax};

#[derive(Debug, Clone, Copy)]
pub enum Metric {
    L1,
    L2,
    LInf,
    LP(f64),
}

impl Metric {
    pub fn new(p: f64) -> Result<Self, KDTreeError> {
        if p.is_infinite() && p.is_sign_positive() {
            return Ok(Self::LInf);
        }
        if !p.is_finite() || p < 1.0 {
            return Err(KDTreeError::InvalidMetric(p));
        }
        if p == 1.0 {
            Ok(Self::L1)
        } else if p == 2.0 {
            Ok(Self::L2)
        } else {
            Ok(Self::LP(p))
        }
    }

    #[inline]
    pub fn finish(self, accum: f64) -> f64 {
        match self {
            Self::L2 => accum.sqrt(),
            Self::LP(p) => accum.powf(1.0 / p),
            Self::L1 | Self::LInf => accum,
        }
    }

    #[inline]
    pub fn to_accum(self, distance: f64) -> f64 {
        match self {
            Self::L2 => distance * distance,
            Self::LP(p) => distance.powf(p),
            Self::L1 | Self::LInf => distance,
        }
    }

    #[inline]
    pub fn eps_factor(self, eps: f64) -> f64 {
        let base = 1.0 + eps;
        match self {
            Self::L2 => base * base,
            Self::LP(p) => base.powf(p),
            Self::L1 | Self::LInf => base,
        }
    }

    /// Accumulate the per-axis contributions of `(lhs - rhs)` and return as
    /// soon as a chunk's running total exceeds `bound`. The accumulator is
    /// monotonically non-decreasing for every supported metric, so the
    /// early-out value is still a valid lower bound that the caller can
    /// compare against `bound` to reject the point.
    ///
    /// The bound check happens once per SIMD chunk, not per axis: a
    /// horizontal reduction every axis would serialize the kernel, while a
    /// full-row scan would give up the early exit that saves most of the
    /// work in high dimensions.
    #[inline]
    pub fn point_accum(self, lhs: &[f64], rhs: &[f64], bound: f64) -> f64 {
        match self {
            Self::L1 | Self::L2 => self.point_accum_sum(lhs, rhs, bound),
            Self::LInf => Self::point_accum_linf(lhs, rhs, bound),
            Self::LP(p) => Self::point_accum_lp(p, lhs, rhs, bound),
        }
    }

    fn point_accum_sum(self, lhs: &[f64], rhs: &[f64], bound: f64) -> f64 {
        let (l_chunks, l_rest) = lhs.as_chunks::<LANES>();
        let (r_chunks, r_rest) = rhs.as_chunks::<LANES>();
        let mut acc = 0.0_f64;
        for (l, r) in l_chunks.iter().zip(r_chunks) {
            acc += hsum(self.axes_lanes(F64s::from_array(*l) - F64s::from_array(*r)));
            if acc > bound {
                return acc;
            }
        }
        self.fold_scalar(acc, l_rest, r_rest)
    }

    /// `L^inf` needs neither a horizontal reduction per chunk nor a scalar
    /// accumulator: the running maximum stays lane-wise, the early exit is a
    /// mask test, and a single horizontal max happens on the way out.
    fn point_accum_linf(lhs: &[f64], rhs: &[f64], bound: f64) -> f64 {
        let (l_chunks, l_rest) = lhs.as_chunks::<LANES>();
        let (r_chunks, r_rest) = rhs.as_chunks::<LANES>();
        if l_chunks.is_empty() {
            return Self::LInf.fold_scalar(0.0, l_rest, r_rest);
        }
        let mut vm = F64s::splat(0.0);
        let bound_v = F64s::splat(bound);
        for (l, r) in l_chunks.iter().zip(r_chunks) {
            vm = vmax(vm, (F64s::from_array(*l) - F64s::from_array(*r)).abs());
            if vm.simd_gt(bound_v).any() {
                return hmax(vm);
            }
        }
        Self::LInf.fold_scalar(hmax(vm), l_rest, r_rest)
    }

    /// `powf` has no SIMD lowering, so `L^p` keeps a scalar loop with the
    /// same chunk-granular early exit as the SIMD paths.
    fn point_accum_lp(p: f64, lhs: &[f64], rhs: &[f64], bound: f64) -> f64 {
        let mut acc = 0.0_f64;
        for (l, r) in lhs.chunks(LANES).zip(rhs.chunks(LANES)) {
            for (a, b) in l.iter().zip(r) {
                acc += (a - b).abs().powf(p);
            }
            if acc > bound {
                return acc;
            }
        }
        acc
    }

    /// Scalar fold over any run of axes. The match stays outside the loop so
    /// each arm keeps a tight body LLVM can unroll or vectorize.
    #[inline(always)]
    fn fold_scalar(self, mut acc: f64, lhs: &[f64], rhs: &[f64]) -> f64 {
        match self {
            Self::L1 => {
                for (a, b) in lhs.iter().zip(rhs) {
                    acc += (a - b).abs();
                }
            }
            Self::L2 => {
                for (a, b) in lhs.iter().zip(rhs) {
                    let delta = a - b;
                    acc += delta * delta;
                }
            }
            Self::LInf => {
                for (a, b) in lhs.iter().zip(rhs) {
                    let delta = (a - b).abs();
                    if delta > acc {
                        acc = delta;
                    }
                }
            }
            Self::LP(_) => unreachable!("LP accumulates only in point_accum_lp"),
        }
        acc
    }

    /// Scan a whole leaf block against `bound`, invoking `on_hit(offset,
    /// accumulated_distance)` for every point within it. `on_hit` returns
    /// the tightened bound to use from that point on.
    ///
    /// Rows of 2-4 dims are too short for `point_accum`'s early exit to save
    /// anything, so they get branch-free kernels that vectorize across
    /// points and keep distances in registers, leaving the SIMD domain only
    /// for the (rare) in-bound candidates. Everything else scans point by
    /// point with the early exit.
    pub fn scan_block(
        self,
        q: &[f64],
        block: &[f64],
        mut bound: f64,
        mut on_hit: impl FnMut(usize, f64) -> f64,
    ) {
        if let Self::LP(_) = self {
            return self.scan_points(q, block, bound, &mut on_hit);
        }
        match q.len() {
            2 => self.scan_block_d2(q, block, &mut bound, &mut on_hit),
            3 => self.scan_rows::<3>(q, block, 0, &mut bound, &mut on_hit),
            4 => self.scan_block_d4(q, block, &mut bound, &mut on_hit),
            _ => self.scan_points(q, block, bound, &mut on_hit),
        }
    }

    /// Per-point scan whose `point_accum` early exit skips most of each row
    /// once the bound tightens.
    fn scan_points(
        self,
        q: &[f64],
        block: &[f64],
        mut bound: f64,
        on_hit: &mut impl FnMut(usize, f64) -> f64,
    ) {
        let ndim = q.len();
        for (j, coords) in block.chunks_exact(ndim).enumerate() {
            let d = self.point_accum(q, coords, bound);
            if d <= bound {
                bound = on_hit(j, d);
            }
        }
    }

    /// Scalar per-point scan for a compile-time row width: the axis loop
    /// fully unrolls, and with no early exit LLVM vectorizes it across
    /// points. Also serves as the sub-register tail of the d2/d4 kernels.
    #[inline(always)]
    fn scan_rows<const D: usize>(
        self,
        q: &[f64],
        rows: &[f64],
        base: usize,
        bound: &mut f64,
        on_hit: &mut impl FnMut(usize, f64) -> f64,
    ) {
        let q = q.first_chunk::<D>().expect("row width exceeds query");
        for (j, p) in rows.as_chunks::<D>().0.iter().enumerate() {
            let d = self.fold_scalar(0.0, q, p);
            if d <= *bound {
                *bound = on_hit(base + j, d);
            }
        }
    }

    /// Four 2-d points per register: lanes hold `[x0 y0 x1 y1 x2 y2 x3 y3]`.
    fn scan_block_d2(
        self,
        q: &[f64],
        block: &[f64],
        bound: &mut f64,
        on_hit: &mut impl FnMut(usize, f64) -> f64,
    ) {
        let pattern = F64s::from_array([q[0], q[1], q[0], q[1], q[0], q[1], q[0], q[1]]);
        let (chunks, rest) = block.as_chunks::<LANES>();
        let mut bound_v = Simd::splat(*bound);
        for (i, b) in chunks.iter().enumerate() {
            let axes = self.axes_lanes(F64s::from_array(*b) - pattern);
            let x = simd_swizzle!(axes, [0, 2, 4, 6]);
            let y = simd_swizzle!(axes, [1, 3, 5, 7]);
            let dists = self.combine(x, y);
            if dists.simd_le(bound_v).any() {
                Self::emit_hits(dists, i * 4, bound, on_hit);
                bound_v = Simd::splat(*bound);
            }
        }
        self.scan_rows::<2>(q, rest, chunks.len() * 4, bound, on_hit);
    }

    /// Two 4-d points per register: lanes hold `[p0: 0..4 | p1: 0..4]`.
    fn scan_block_d4(
        self,
        q: &[f64],
        block: &[f64],
        bound: &mut f64,
        on_hit: &mut impl FnMut(usize, f64) -> f64,
    ) {
        let pattern = F64s::from_array([q[0], q[1], q[2], q[3], q[0], q[1], q[2], q[3]]);
        let (chunks, rest) = block.as_chunks::<LANES>();
        let mut bound_v = Simd::splat(*bound);
        for (i, b) in chunks.iter().enumerate() {
            let axes = self.axes_lanes(F64s::from_array(*b) - pattern);
            let pairs = self.combine(
                simd_swizzle!(axes, [0, 1, 4, 5]),
                simd_swizzle!(axes, [2, 3, 6, 7]),
            );
            let dists = self.combine(simd_swizzle!(pairs, [0, 2]), simd_swizzle!(pairs, [1, 3]));
            if dists.simd_le(bound_v).any() {
                Self::emit_hits(dists, i * 2, bound, on_hit);
                bound_v = Simd::splat(*bound);
            }
        }
        self.scan_rows::<4>(q, rest, chunks.len() * 2, bound, on_hit);
    }

    /// Fan the in-bound lanes of a distance vector out to `on_hit`. Callers
    /// re-splat their vector bound afterwards, since a hit tightens it.
    #[inline(always)]
    fn emit_hits<const N: usize>(
        dists: Simd<f64, N>,
        base: usize,
        bound: &mut f64,
        on_hit: &mut impl FnMut(usize, f64) -> f64,
    ) {
        for (j, &d) in dists.as_array().iter().enumerate() {
            if d <= *bound {
                *bound = on_hit(base + j, d);
            }
        }
    }

    /// Per-axis contribution of a lane-wise difference.
    #[inline(always)]
    fn axes_lanes(self, delta: F64s) -> F64s {
        match self {
            Self::L1 | Self::LInf => delta.abs(),
            Self::L2 => delta * delta,
            Self::LP(_) => unreachable!("LP is routed to scan_points"),
        }
    }

    /// Combine two vectors of partial accumulations lane-wise: max for
    /// `L^inf`, sum for every other metric.
    #[inline(always)]
    fn combine<const N: usize>(self, a: Simd<f64, N>, b: Simd<f64, N>) -> Simd<f64, N> {
        match self {
            Self::LInf => vmax(a, b),
            _ => a + b,
        }
    }

    /// Per-axis contribution of a non-negative axis offset.
    #[inline]
    pub fn axis_accum(self, offset: f64) -> f64 {
        debug_assert!(offset >= 0.0);
        match self {
            Self::L1 | Self::LInf => offset,
            Self::L2 => offset * offset,
            Self::LP(p) => offset.powf(p),
        }
    }

    /// Fold a per-axis contribution into a running accumulator. Sum for
    /// L^p, max for L^inf.
    #[inline]
    pub fn fold_axis(self, acc: f64, axis: f64) -> f64 {
        match self {
            Self::LInf => acc.max(axis),
            _ => acc + axis,
        }
    }

    /// Update an accumulator when a single axis's contribution changes from
    /// `old_axis` to `new_axis`. The caller must guarantee
    /// `new_axis >= old_axis`, which is always the case when descending from
    /// a parent cell into the far child along the split.
    #[inline]
    pub fn replace_axis(self, total: f64, old_axis: f64, new_axis: f64) -> f64 {
        match self {
            Self::LInf => total.max(new_axis),
            _ => total - old_axis + new_axis,
        }
    }

    /// Accumulated distance from `q` to the axis-aligned box `[lo, hi]`:
    /// zero when `q` is inside. Comparable against the same accumulator
    /// domain as `point_accum` (squared for L2, p-th power for L^p).
    #[inline]
    pub fn bbox_accum(self, q: &[f64], lo: &[f64], hi: &[f64]) -> f64 {
        match self {
            Self::L1 | Self::L2 => self.bbox_accum_sum(q, lo, hi),
            Self::LInf => Self::bbox_accum_linf(q, lo, hi),
            Self::LP(p) => Self::bbox_accum_lp(p, q, lo, hi),
        }
    }

    fn bbox_accum_sum(self, q: &[f64], lo: &[f64], hi: &[f64]) -> f64 {
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
            acc += self.axes_lanes(off);
        }
        let mut total = if q_chunks.is_empty() { 0.0 } else { hsum(acc) };
        for ((&qs, &ls), &hs) in q_rest.iter().zip(lo_rest).zip(hi_rest) {
            let off = (ls - qs).max(qs - hs).max(0.0);
            total += match self {
                Self::L2 => off * off,
                _ => off,
            };
        }
        total
    }

    fn bbox_accum_linf(q: &[f64], lo: &[f64], hi: &[f64]) -> f64 {
        let mut worst = 0.0_f64;
        for ((&qs, &ls), &hs) in q.iter().zip(lo).zip(hi) {
            let off = (ls - qs).max(qs - hs);
            if off > worst {
                worst = off;
            }
        }
        worst
    }

    fn bbox_accum_lp(p: f64, q: &[f64], lo: &[f64], hi: &[f64]) -> f64 {
        let mut total = 0.0_f64;
        for ((&qs, &ls), &hs) in q.iter().zip(lo).zip(hi) {
            let off = (ls - qs).max(qs - hs).max(0.0);
            if off > 0.0 {
                total += off.powf(p);
            }
        }
        total
    }
}
