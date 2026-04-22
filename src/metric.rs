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
        if (p - 1.0).abs() < f64::EPSILON {
            Ok(Self::L1)
        } else if (p - 2.0).abs() < f64::EPSILON {
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
    /// We bound-check at chunk boundaries instead of every axis so LLVM can
    /// still auto-vectorize the inner accumulation; per-axis branching kills
    /// SIMD and is a net loss even for moderate `ndim`.
    #[inline]
    pub fn point_accum(self, lhs: &[f64], rhs: &[f64], bound: f64) -> f64 {
        const CHUNK: usize = 8;
        let mut acc = 0.0_f64;
        let mut lhs_rest = lhs;
        let mut rhs_rest = rhs;
        while lhs_rest.len() >= CHUNK {
            let (l_head, l_tail) = lhs_rest.split_at(CHUNK);
            let (r_head, r_tail) = rhs_rest.split_at(CHUNK);
            acc = self.fold_block(acc, l_head, r_head);
            if acc > bound {
                return acc;
            }
            lhs_rest = l_tail;
            rhs_rest = r_tail;
        }
        self.fold_block(acc, lhs_rest, rhs_rest)
    }

    #[inline(always)]
    fn fold_block(self, mut acc: f64, lhs: &[f64], rhs: &[f64]) -> f64 {
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
            Self::LP(p) => {
                for (a, b) in lhs.iter().zip(rhs) {
                    acc += (a - b).abs().powf(p);
                }
            }
        }
        acc
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
