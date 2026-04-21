#[derive(Debug, Clone)]
pub struct Node {
    pub start: usize,
    pub end: usize,
    pub split_dim: usize,
    pub split_value: f64,
    pub left: Option<usize>,
    pub right: Option<usize>,
    pub leaf: bool,
}
