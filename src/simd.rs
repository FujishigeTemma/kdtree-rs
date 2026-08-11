//! Small portable-SIMD building blocks shared by the distance kernels and
//! the tree build. Everything here is architecture-neutral `std::simd`.

use std::simd::prelude::*;
use std::simd::simd_swizzle;

/// Lane count shared by every SIMD kernel in the crate. Eight f64 lanes is a
/// logical width: `std::simd` lowers it to whatever the target provides
/// (2 lanes per op on SSE2, 4 on AVX2/NEON pairs, 8 on AVX-512) with no
/// architecture-specific code on our side. The swizzle patterns in `hsum`,
/// `hmax`, and the block kernels hardcode eight lanes, so this constant
/// names the width rather than tuning it.
pub(crate) const LANES: usize = 8;
pub(crate) type F64s = Simd<f64, LANES>;

/// Lane-wise max via compare+select. All data in this crate is validated
/// finite on construction, so IEEE NaN semantics are irrelevant and the
/// compare form lowers to a single packed max instruction, where
/// `SimdFloat::simd_max`'s maxNum semantics can require fixup sequences.
#[inline(always)]
pub(crate) fn vmax<const N: usize>(a: Simd<f64, N>, b: Simd<f64, N>) -> Simd<f64, N> {
    b.simd_gt(a).select(b, a)
}

/// Lane-wise min; see `vmax` for why this beats `simd_min` here.
#[inline(always)]
pub(crate) fn vmin<const N: usize>(a: Simd<f64, N>, b: Simd<f64, N>) -> Simd<f64, N> {
    b.simd_lt(a).select(b, a)
}

/// Horizontal tree sum of all lanes. `SimdFloat::reduce_sum` lowers to an
/// ordered (fully serialized) reduction; pairwise packed adds are both
/// faster and deterministic across targets.
#[inline(always)]
pub(crate) fn hsum(v: F64s) -> f64 {
    let s4 = simd_swizzle!(v, [0, 1, 2, 3]) + simd_swizzle!(v, [4, 5, 6, 7]);
    let s2 = simd_swizzle!(s4, [0, 1]) + simd_swizzle!(s4, [2, 3]);
    s2[0] + s2[1]
}

/// Horizontal max of all lanes. `SimdFloat::reduce_max` falls back to a
/// scalar NaN-propagating fold, so the tree is built from `vmax` instead.
#[inline(always)]
pub(crate) fn hmax(v: F64s) -> f64 {
    let m4 = vmax(simd_swizzle!(v, [0, 1, 2, 3]), simd_swizzle!(v, [4, 5, 6, 7]));
    let m2 = vmax(simd_swizzle!(m4, [0, 1]), simd_swizzle!(m4, [2, 3]));
    if m2[1] > m2[0] { m2[1] } else { m2[0] }
}

/// Vectorized finiteness sweep: `v * 0 != 0` exactly for infinities and
/// NaNs, so one multiply-compare per vector replaces per-element
/// `is_finite` branches.
pub(crate) fn all_finite(values: &[f64]) -> bool {
    let zero = F64s::splat(0.0);
    let mut nonfinite = zero.simd_ne(zero);
    let (chunks, rest) = values.as_chunks::<LANES>();
    for c in chunks {
        nonfinite |= (F64s::from_array(*c) * zero).simd_ne(zero);
    }
    !nonfinite.any() && rest.iter().all(|v| v.is_finite())
}
