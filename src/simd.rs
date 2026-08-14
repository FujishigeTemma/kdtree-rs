use std::simd::prelude::*;
use std::simd::simd_swizzle;

/// A logical width `std::simd` lowers to whatever the target has. Not a tuning
/// knob — the swizzle patterns here and in `kernel.rs` hardcode eight lanes.
pub(crate) const LANES: usize = 8;
pub(crate) type F64s = Simd<f64, LANES>;

/// All coordinates are validated finite at construction, so NaN semantics never
/// arise and compare+select lowers to one packed max. `SimdFloat::simd_max`'s
/// maxNum semantics can require fixup sequences instead.
#[inline(always)]
pub(crate) fn vmax<const N: usize>(a: Simd<f64, N>, b: Simd<f64, N>) -> Simd<f64, N> {
    b.simd_gt(a).select(b, a)
}

/// See [`vmax`].
#[inline(always)]
pub(crate) fn vmin<const N: usize>(a: Simd<f64, N>, b: Simd<f64, N>) -> Simd<f64, N> {
    b.simd_lt(a).select(b, a)
}

/// Pairwise tree rather than `SimdFloat::reduce_sum`, which lowers to an ordered
/// (fully serialized) reduction.
#[inline(always)]
pub(crate) fn hsum(v: F64s) -> f64 {
    let s4 = simd_swizzle!(v, [0, 1, 2, 3]) + simd_swizzle!(v, [4, 5, 6, 7]);
    let s2 = simd_swizzle!(s4, [0, 1]) + simd_swizzle!(s4, [2, 3]);
    s2[0] + s2[1]
}

/// Pairwise tree rather than `SimdFloat::reduce_max`, which falls back to a
/// scalar NaN-propagating fold.
#[inline(always)]
pub(crate) fn hmax(v: F64s) -> f64 {
    let m4 = vmax(
        simd_swizzle!(v, [0, 1, 2, 3]),
        simd_swizzle!(v, [4, 5, 6, 7]),
    );
    let m2 = vmax(simd_swizzle!(m4, [0, 1]), simd_swizzle!(m4, [2, 3]));
    if m2[1] > m2[0] { m2[1] } else { m2[0] }
}

/// `v * 0 != 0` holds exactly for infinities and NaNs, so one multiply-compare
/// per vector replaces per-element `is_finite`.
#[inline(always)]
pub(crate) fn nonfinite_lanes(v: F64s) -> Mask<i64, LANES> {
    let zero = F64s::splat(0.0);
    (v * zero).simd_ne(zero)
}
