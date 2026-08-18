use crate::error::KDTreeError;

/// A distance as the crate's core handles it. Its representation belongs to the
/// producing [`Metric`] — squared for `L2`, the p-th power for `L^p`, the plain
/// value elsewhere — a monotone image, so every comparison and accumulation
/// stays in it and the root is taken once per emitted result rather than once
/// per candidate. Raw `f64` distances exist only at the API boundary, crossing
/// through [`Metric::reduce`] / [`Metric::restore`]. Only meaningfully ordered
/// against values of the same metric.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[repr(transparent)]
pub(crate) struct Dist(f64);

impl Dist {
    pub(crate) const ZERO: Self = Self(0.0);
    pub(crate) const INFINITY: Self = Self(f64::INFINITY);

    /// Wrap a value already in the metric's representation — for kernels that
    /// fold raw SIMD lanes. Named so every such site is greppable.
    #[inline(always)]
    pub(crate) const fn from_repr(value: f64) -> Self {
        Self(value)
    }

    #[inline(always)]
    pub(crate) const fn get(self) -> f64 {
        self.0
    }

    #[inline(always)]
    pub(crate) fn min(self, other: Self) -> Self {
        Self(self.0.min(other.0))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Metric {
    L1,
    L2,
    LInf,
    LP(f64),
}

impl Metric {
    pub(crate) fn new(p: f64) -> Result<Self, KDTreeError> {
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

    /// Takes a whole distance or a single axis offset — one axis reduces exactly
    /// like a whole distance.
    #[inline]
    pub(crate) fn reduce(self, distance: f64) -> Dist {
        debug_assert!(distance >= 0.0);
        Dist(match self {
            Self::L2 => distance * distance,
            Self::LP(p) => distance.powf(p),
            Self::L1 | Self::LInf => distance,
        })
    }

    #[inline]
    pub(crate) fn restore(self, dist: Dist) -> f64 {
        match self {
            Self::L2 => dist.0.sqrt(),
            Self::LP(p) => dist.0.powf(1.0 / p),
            Self::L1 | Self::LInf => dist.0,
        }
    }

    #[inline]
    pub(crate) fn fold(self, dist: Dist, axis: Dist) -> Dist {
        Dist(match self {
            Self::LInf => dist.0.max(axis.0),
            _ => dist.0 + axis.0,
        })
    }

    /// Requires `new >= old`, which holds whenever a descent moves from a parent
    /// cell into the far child along the split.
    #[inline]
    pub(crate) fn replace_axis(self, dist: Dist, old: Dist, new: Dist) -> Dist {
        Dist(match self {
            Self::LInf => dist.0.max(new.0),
            _ => dist.0 - old.0 + new.0,
        })
    }
}
