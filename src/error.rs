use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum KDTreeError {
    EmptyData,
    InvalidLeafsize,
    InvalidK,
    InvalidRadius(f64),
    InvalidMetric(f64),
    InvalidEps(f64),
    InvalidMaxDistance(f64),
    NonFiniteData,
    InvalidShape(&'static str),
    DimensionMismatch { expected: usize, got: usize },
}

impl fmt::Display for KDTreeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyData => write!(f, "data must contain at least one point"),
            Self::InvalidLeafsize => write!(f, "leafsize must be greater than zero"),
            Self::InvalidK => write!(f, "k must be greater than zero"),
            Self::InvalidRadius(radius) => {
                write!(f, "radius must be finite and non-negative, got {radius}")
            }
            Self::InvalidMetric(p) => {
                write!(f, "p must be finite and >= 1, or infinity, got {p}")
            }
            Self::InvalidEps(eps) => write!(f, "eps must be finite and non-negative, got {eps}"),
            Self::InvalidMaxDistance(distance) => {
                write!(
                    f,
                    "max_distance must be finite and non-negative, or infinity, got {distance}"
                )
            }
            Self::NonFiniteData => write!(f, "all coordinates must be finite"),
            Self::InvalidShape(message) => write!(f, "{message}"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
        }
    }
}

impl std::error::Error for KDTreeError {}
