use crate::error::KDTreeError;

/// A distance in the reduced (monotone-image) domain of some [`Metric`]:
/// squared for `L2`, the p-th power for `L^p`, the plain value elsewhere. Every
/// comparison and accumulation in the crate stays here, so the root is taken once
/// per emitted result rather than once per candidate. Only meaningfully ordered
/// against values of the same metric.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[repr(transparent)]
pub(crate) struct Rd(f64);

impl Rd {
    pub(crate) const ZERO: Self = Self(0.0);
    pub(crate) const INFINITY: Self = Self(f64::INFINITY);

    /// Re-enter the domain with a value already in it — for kernels that fold
    /// raw SIMD lanes. Named so every such site is greppable.
    #[inline(always)]
    pub(crate) const fn reduced(value: f64) -> Self {
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

    /// Scale by a factor that is itself already reduced ([`Metric::eps_factor`]).
    #[inline(always)]
    pub(crate) fn scaled(self, factor: f64) -> Self {
        Self(self.0 * factor)
    }
}

#[derive(Debug, Clone, Copy)]
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
    pub(crate) fn reduce(self, distance: f64) -> Rd {
        debug_assert!(distance >= 0.0);
        Rd(match self {
            Self::L2 => distance * distance,
            Self::LP(p) => distance.powf(p),
            Self::L1 | Self::LInf => distance,
        })
    }

    #[inline]
    pub(crate) fn restore(self, rd: Rd) -> f64 {
        match self {
            Self::L2 => rd.0.sqrt(),
            Self::LP(p) => rd.0.powf(1.0 / p),
            Self::L1 | Self::LInf => rd.0,
        }
    }

    #[inline]
    pub(crate) fn eps_factor(self, eps: f64) -> f64 {
        self.reduce(1.0 + eps).0
    }

    #[inline]
    pub(crate) fn fold(self, rd: Rd, axis: Rd) -> Rd {
        Rd(match self {
            Self::LInf => rd.0.max(axis.0),
            _ => rd.0 + axis.0,
        })
    }

    /// Requires `new >= old`, which holds whenever a descent moves from a parent
    /// cell into the far child along the split.
    #[inline]
    pub(crate) fn replace_axis(self, rd: Rd, old: Rd, new: Rd) -> Rd {
        Rd(match self {
            Self::LInf => rd.0.max(new.0),
            _ => rd.0 - old.0 + new.0,
        })
    }
}
