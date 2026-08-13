//! The `L^p` metric algebra over *reduced distances*.
//!
//! A reduced distance (`rd` throughout the crate) is a metric-specific
//! monotone image of the true distance: squared for `L2`, the p-th power
//! for `L^p`, the plain value for `L1`/`L^inf`. Comparisons, pruning
//! bounds, and accumulation all stay in this domain, so the root is taken
//! once per emitted result instead of once per candidate. This module owns
//! the per-axis algebra of that domain; the vectorized loops that apply it
//! to points, leaf blocks, and bounding boxes live in `kernel.rs`.

use crate::error::KDTreeError;

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

    /// True distance -> reduced distance.
    #[inline]
    pub fn reduce(self, distance: f64) -> f64 {
        match self {
            Self::L2 => distance * distance,
            Self::LP(p) => distance.powf(p),
            Self::L1 | Self::LInf => distance,
        }
    }

    /// Reduced distance -> true distance; the inverse of [`Metric::reduce`].
    #[inline]
    pub fn restore(self, rd: f64) -> f64 {
        match self {
            Self::L2 => rd.sqrt(),
            Self::LP(p) => rd.powf(1.0 / p),
            Self::L1 | Self::LInf => rd,
        }
    }

    /// `(1 + eps)` carried into the reduced domain, so approximate-search
    /// pruning multiplies lower bounds instead of re-rooting distances.
    #[inline]
    pub fn eps_factor(self, eps: f64) -> f64 {
        self.reduce(1.0 + eps)
    }

    /// Reduced contribution of one axis, from a non-negative offset. A single
    /// axis reduces exactly like a whole distance; the separate name marks
    /// the per-axis call sites.
    #[inline]
    pub fn axis_rd(self, offset: f64) -> f64 {
        debug_assert!(offset >= 0.0);
        self.reduce(offset)
    }

    /// Fold one axis contribution into a running reduced distance: sum for
    /// `L^p`, max for `L^inf`.
    #[inline]
    pub fn fold(self, rd: f64, axis: f64) -> f64 {
        match self {
            Self::LInf => rd.max(axis),
            _ => rd + axis,
        }
    }

    /// Reduced distance after one axis's contribution changes from
    /// `old_axis` to `new_axis`. The caller must guarantee
    /// `new_axis >= old_axis`, which always holds when descending from a
    /// parent cell into the far child along the split.
    #[inline]
    pub fn replace_axis(self, rd: f64, old_axis: f64, new_axis: f64) -> f64 {
        match self {
            Self::LInf => rd.max(new_axis),
            _ => rd - old_axis + new_axis,
        }
    }
}

/// Per-axis offset from a coordinate to the interval `[lo, hi]`: zero
/// inside. This is the definition every box bound in the crate computes;
/// `kernel.rs`'s vector paths inline the same `max(lo - q, q - hi, 0)` with
/// lane-wise maxima rather than calling it.
#[inline(always)]
pub fn box_axis_offset(q: f64, lo: f64, hi: f64) -> f64 {
    (lo - q).max(q - hi).max(0.0)
}
