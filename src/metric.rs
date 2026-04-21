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

    #[inline]
    pub fn point_accum(self, lhs: &[f64], rhs: &[f64]) -> f64 {
        match self {
            Self::L1 => lhs.iter().zip(rhs).map(|(a, b)| (a - b).abs()).sum(),
            Self::L2 => lhs
                .iter()
                .zip(rhs)
                .map(|(a, b)| {
                    let delta = a - b;
                    delta * delta
                })
                .sum(),
            Self::LInf => lhs
                .iter()
                .zip(rhs)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max),
            Self::LP(p) => lhs
                .iter()
                .zip(rhs)
                .map(|(a, b)| (a - b).abs().powf(p))
                .sum(),
        }
    }

    #[inline]
    pub fn bbox_point_lower_bound(self, query: &[f64], mins: &[f64], maxes: &[f64]) -> f64 {
        match self {
            Self::L1 => query
                .iter()
                .zip(mins.iter().zip(maxes))
                .map(|(q, (min, max))| distance_to_interval(*q, *min, *max))
                .sum(),
            Self::L2 => query
                .iter()
                .zip(mins.iter().zip(maxes))
                .map(|(q, (min, max))| {
                    let delta = distance_to_interval(*q, *min, *max);
                    delta * delta
                })
                .sum(),
            Self::LInf => query
                .iter()
                .zip(mins.iter().zip(maxes))
                .map(|(q, (min, max))| distance_to_interval(*q, *min, *max))
                .fold(0.0, f64::max),
            Self::LP(p) => query
                .iter()
                .zip(mins.iter().zip(maxes))
                .map(|(q, (min, max))| distance_to_interval(*q, *min, *max).powf(p))
                .sum(),
        }
    }

    #[inline]
    pub fn bbox_bbox_lower_bound(
        self,
        mins_a: &[f64],
        maxes_a: &[f64],
        mins_b: &[f64],
        maxes_b: &[f64],
    ) -> f64 {
        match self {
            Self::L1 => mins_a
                .iter()
                .zip(maxes_a)
                .zip(mins_b.iter().zip(maxes_b))
                .map(|((min_a, max_a), (min_b, max_b))| interval_gap(*min_a, *max_a, *min_b, *max_b))
                .sum(),
            Self::L2 => mins_a
                .iter()
                .zip(maxes_a)
                .zip(mins_b.iter().zip(maxes_b))
                .map(|((min_a, max_a), (min_b, max_b))| {
                    let gap = interval_gap(*min_a, *max_a, *min_b, *max_b);
                    gap * gap
                })
                .sum(),
            Self::LInf => mins_a
                .iter()
                .zip(maxes_a)
                .zip(mins_b.iter().zip(maxes_b))
                .map(|((min_a, max_a), (min_b, max_b))| interval_gap(*min_a, *max_a, *min_b, *max_b))
                .fold(0.0, f64::max),
            Self::LP(p) => mins_a
                .iter()
                .zip(maxes_a)
                .zip(mins_b.iter().zip(maxes_b))
                .map(|((min_a, max_a), (min_b, max_b))| interval_gap(*min_a, *max_a, *min_b, *max_b).powf(p))
                .sum(),
        }
    }
}

#[inline]
fn distance_to_interval(value: f64, min: f64, max: f64) -> f64 {
    if value < min {
        min - value
    } else if value > max {
        value - max
    } else {
        0.0
    }
}

#[inline]
fn interval_gap(min_a: f64, max_a: f64, min_b: f64, max_b: f64) -> f64 {
    if max_a < min_b {
        min_b - max_a
    } else if max_b < min_a {
        min_a - max_b
    } else {
        0.0
    }
}
