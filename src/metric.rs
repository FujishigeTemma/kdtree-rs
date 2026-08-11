use std::simd::prelude::*;
use std::simd::simd_swizzle;

use crate::error::KDTreeError;

/// Lane count shared by every SIMD kernel in the crate. Eight f64 lanes is a
/// logical width: `std::simd` lowers it to whatever the target provides
/// (2 lanes per op on SSE2, 4 on AVX2/NEON pairs, 8 on AVX-512) with no
/// architecture-specific code on our side.
pub(crate) const LANES: usize = 8;
pub(crate) type F64s = Simd<f64, LANES>;

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
            Self::LP(_) => return self.point_accum_scalar(lhs, rhs, bound),
            Self::LInf => return Self::point_accum_linf(lhs, rhs, bound),
            Self::L1 | Self::L2 => {}
        }
        let (l_chunks, l_rest) = lhs.as_chunks::<LANES>();
        let (r_chunks, r_rest) = rhs.as_chunks::<LANES>();
        let mut acc = 0.0_f64;
        for (l, r) in l_chunks.iter().zip(r_chunks) {
            acc = self.fold_lanes(acc, F64s::from_array(*l), F64s::from_array(*r));
            if acc > bound {
                return acc;
            }
        }
        self.fold_tail(acc, l_rest, r_rest)
    }

    /// `powf` has no SIMD lowering, so `L^p` keeps a scalar loop with the
    /// same chunk-granular early exit as the SIMD path.
    fn point_accum_scalar(self, lhs: &[f64], rhs: &[f64], bound: f64) -> f64 {
        let mut acc = 0.0_f64;
        for (l, r) in lhs.chunks(LANES).zip(rhs.chunks(LANES)) {
            acc = self.fold_tail(acc, l, r);
            if acc > bound {
                return acc;
            }
        }
        acc
    }

    /// Fold one SIMD chunk of per-axis contributions into `acc` (sum
    /// metrics only; `L^inf` and `L^p` never reach this).
    #[inline(always)]
    fn fold_lanes(self, acc: f64, lhs: F64s, rhs: F64s) -> f64 {
        acc + hsum(self.axes_lanes(lhs - rhs))
    }

    /// `L^inf` needs neither a horizontal reduction per chunk nor a scalar
    /// accumulator: the running maximum stays lane-wise, the early exit is a
    /// mask test, and a single horizontal max happens on the way out.
    fn point_accum_linf(lhs: &[f64], rhs: &[f64], bound: f64) -> f64 {
        let (l_chunks, l_rest) = lhs.as_chunks::<LANES>();
        let (r_chunks, r_rest) = rhs.as_chunks::<LANES>();
        let mut vmax = F64s::splat(0.0);
        let bound_v = F64s::splat(bound);
        for (l, r) in l_chunks.iter().zip(r_chunks) {
            let axes = (F64s::from_array(*l) - F64s::from_array(*r)).abs();
            vmax = axes.simd_gt(vmax).select(axes, vmax);
            if axes.simd_gt(bound_v).any() {
                return hmax(vmax);
            }
        }
        let mut acc = hmax(vmax);
        for (a, b) in l_rest.iter().zip(r_rest) {
            let delta = (a - b).abs();
            if delta > acc {
                acc = delta;
            }
        }
        acc
    }

    /// Scalar fold for slices shorter than one SIMD chunk.
    #[inline(always)]
    fn fold_tail(self, mut acc: f64, lhs: &[f64], rhs: &[f64]) -> f64 {
        for (a, b) in lhs.iter().zip(rhs) {
            match self {
                Self::L1 => acc += (a - b).abs(),
                Self::L2 => {
                    let delta = a - b;
                    acc += delta * delta;
                }
                Self::LInf => {
                    let delta = (a - b).abs();
                    if delta > acc {
                        acc = delta;
                    }
                }
                Self::LP(p) => acc += (a - b).abs().powf(p),
            }
        }
        acc
    }

    /// Whether `scan_block` has a kernel for this metric/dimensionality.
    /// Below ~5 dimensions a row is too short for the per-point early exit
    /// to save anything, so the block kernels vectorize across points
    /// instead of across dimensions.
    #[inline]
    pub fn has_block_kernel(self, ndim: usize) -> bool {
        !matches!(self, Self::LP(_)) && matches!(ndim, 2..=4)
    }

    /// Scan a whole leaf block against `bound`, invoking `on_hit(offset,
    /// accumulated_distance)` for every point within it. `on_hit` returns
    /// the tightened bound to use from that point on. Distances stay in
    /// registers: candidates are detected with a lane-wise compare and only
    /// the (rare) hits leave the SIMD domain.
    pub fn scan_block(
        self,
        q: &[f64],
        block: &[f64],
        bound: f64,
        on_hit: impl FnMut(usize, f64) -> f64,
    ) {
        match q.len() {
            2 => self.scan_block_d2(q, block, bound, on_hit),
            3 => self.scan_block_d3(q, block, bound, on_hit),
            4 => self.scan_block_d4(q, block, bound, on_hit),
            _ => unreachable!("guarded by has_block_kernel"),
        }
    }

    /// Four 2-d points per register: lanes hold `[x0 y0 x1 y1 x2 y2 x3 y3]`.
    fn scan_block_d2(
        self,
        q: &[f64],
        block: &[f64],
        mut bound: f64,
        mut on_hit: impl FnMut(usize, f64) -> f64,
    ) {
        let pattern = F64s::from_array([q[0], q[1], q[0], q[1], q[0], q[1], q[0], q[1]]);
        let (chunks, rest) = block.as_chunks::<LANES>();
        let mut base = 0;
        for b in chunks {
            let axes = self.axes_lanes(F64s::from_array(*b) - pattern);
            let x = simd_swizzle!(axes, [0, 2, 4, 6]);
            let y = simd_swizzle!(axes, [1, 3, 5, 7]);
            let dists = self.combine(x, y);
            if dists.simd_le(Simd::splat(bound)).any() {
                for (j, &d) in dists.as_array().iter().enumerate() {
                    if d <= bound {
                        bound = on_hit(base + j, d);
                    }
                }
            }
            base += 4;
        }
        for (j, p) in rest.as_chunks::<2>().0.iter().enumerate() {
            let d = self.fold_tail(0.0, q, p);
            if d <= bound {
                bound = on_hit(base + j, d);
            }
        }
    }

    /// A 3-wide row straddles register boundaries, so points are evaluated
    /// one at a time with the three axis contributions kept independent;
    /// this vectorizes across points without any shuffle gymnastics.
    fn scan_block_d3(
        self,
        q: &[f64],
        block: &[f64],
        mut bound: f64,
        mut on_hit: impl FnMut(usize, f64) -> f64,
    ) {
        let (q0, q1, q2) = (q[0], q[1], q[2]);
        for (j, p) in block.as_chunks::<3>().0.iter().enumerate() {
            let a0 = self.axis_accum(p[0] - q0);
            let a1 = self.axis_accum(p[1] - q1);
            let a2 = self.axis_accum(p[2] - q2);
            let d = self.fold_axis(self.fold_axis(a0, a1), a2);
            if d <= bound {
                bound = on_hit(j, d);
            }
        }
    }

    /// Two 4-d points per register: lanes hold `[p0: 0..4 | p1: 0..4]`.
    fn scan_block_d4(
        self,
        q: &[f64],
        block: &[f64],
        mut bound: f64,
        mut on_hit: impl FnMut(usize, f64) -> f64,
    ) {
        let pattern = F64s::from_array([q[0], q[1], q[2], q[3], q[0], q[1], q[2], q[3]]);
        let (chunks, rest) = block.as_chunks::<LANES>();
        let mut base = 0;
        for b in chunks {
            let axes = self.axes_lanes(F64s::from_array(*b) - pattern);
            let lo = simd_swizzle!(axes, [0, 1, 4, 5]);
            let hi = simd_swizzle!(axes, [2, 3, 6, 7]);
            let pairs = self.combine(lo, hi);
            let even = simd_swizzle!(pairs, [0, 2]);
            let odd = simd_swizzle!(pairs, [1, 3]);
            let dists = self.combine(even, odd);
            if dists.simd_le(Simd::splat(bound)).any() {
                for (j, &d) in dists.as_array().iter().enumerate() {
                    if d <= bound {
                        bound = on_hit(base + j, d);
                    }
                }
            }
            base += 2;
        }
        for (j, p) in rest.as_chunks::<4>().0.iter().enumerate() {
            let d = self.fold_tail(0.0, q, p);
            if d <= bound {
                bound = on_hit(base + j, d);
            }
        }
    }

    /// Per-axis contribution of a lane-wise difference.
    #[inline(always)]
    fn axes_lanes(self, delta: F64s) -> F64s {
        match self {
            Self::L1 | Self::LInf => delta.abs(),
            Self::L2 => delta * delta,
            Self::LP(_) => unreachable!("LP never takes the batch path"),
        }
    }

    /// Combine two vectors of partial accumulations lane-wise: max for
    /// `L^inf`, sum for every other metric.
    #[inline(always)]
    fn combine<const N: usize>(self, a: Simd<f64, N>, b: Simd<f64, N>) -> Simd<f64, N> {
        match self {
            Self::LInf => b.simd_gt(a).select(b, a),
            _ => a + b,
        }
    }

    #[inline]
    pub fn axis_accum(self, diff: f64) -> f64 {
        match self {
            Self::L1 | Self::LInf => diff.abs(),
            Self::L2 => {
                let a = diff.abs();
                a * a
            }
            Self::LP(p) => diff.abs().powf(p),
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
}

/// Horizontal tree sum of all lanes. `SimdFloat::reduce_sum` lowers to an
/// ordered (fully serialized) reduction; pairwise packed adds are both
/// faster and deterministic across targets.
#[inline(always)]
fn hsum(v: F64s) -> f64 {
    let lo: Simd<f64, 4> = simd_swizzle!(v, [0, 1, 2, 3]);
    let hi: Simd<f64, 4> = simd_swizzle!(v, [4, 5, 6, 7]);
    let s4 = lo + hi;
    let lo2: Simd<f64, 2> = simd_swizzle!(s4, [0, 1]);
    let hi2: Simd<f64, 2> = simd_swizzle!(s4, [2, 3]);
    let s2 = lo2 + hi2;
    s2[0] + s2[1]
}

/// Horizontal max of all lanes. `SimdFloat::reduce_max` falls back to a
/// scalar NaN-propagating fold; the compare-select tree below assumes the
/// finite data this crate validates on construction.
#[inline(always)]
fn hmax(v: F64s) -> f64 {
    let lo: Simd<f64, 4> = simd_swizzle!(v, [0, 1, 2, 3]);
    let hi: Simd<f64, 4> = simd_swizzle!(v, [4, 5, 6, 7]);
    let m4 = hi.simd_gt(lo).select(hi, lo);
    let lo2: Simd<f64, 2> = simd_swizzle!(m4, [0, 1]);
    let hi2: Simd<f64, 2> = simd_swizzle!(m4, [2, 3]);
    let m2 = hi2.simd_gt(lo2).select(hi2, lo2);
    if m2[1] > m2[0] { m2[1] } else { m2[0] }
}
