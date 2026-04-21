use ndarray::ArrayView2;

use crate::data::point;
use crate::error::KDTreeError;
use crate::node::Node;

pub struct Tree {
    data: Vec<f64>,
    indices: Vec<usize>,
    nodes: Vec<Node>,
    bbox_mins: Vec<f64>,
    bbox_maxes: Vec<f64>,
    root: usize,
    n_points: usize,
    ndim: usize,
    leafsize: usize,
}

impl Tree {
    pub fn new(data: ArrayView2<'_, f64>, leafsize: usize) -> Result<Self, KDTreeError> {
        if leafsize == 0 {
            return Err(KDTreeError::InvalidLeafsize);
        }
        if data.nrows() == 0 || data.ncols() == 0 {
            return Err(KDTreeError::EmptyData);
        }
        if !data.iter().all(|value| value.is_finite()) {
            return Err(KDTreeError::NonFiniteData);
        }

        let n_points = data.nrows();
        let ndim = data.ncols();
        let flattened = data.iter().copied().collect::<Vec<_>>();
        let indices = (0..n_points).collect::<Vec<_>>();

        let mut tree = Self {
            data: flattened,
            indices,
            nodes: Vec::with_capacity(n_points.saturating_mul(2)),
            bbox_mins: Vec::new(),
            bbox_maxes: Vec::new(),
            root: 0,
            n_points,
            ndim,
            leafsize,
        };
        let root = tree.build_node(0, n_points);
        tree.root = root;
        Ok(tree)
    }

    pub fn data(&self) -> &[f64] {
        &self.data
    }

    pub fn ndim(&self) -> usize {
        self.ndim
    }

    pub fn n_points(&self) -> usize {
        self.n_points
    }

    pub fn leafsize(&self) -> usize {
        self.leafsize
    }

    pub(crate) fn root(&self) -> usize {
        self.root
    }

    pub(crate) fn node(&self, index: usize) -> &Node {
        &self.nodes[index]
    }

    pub(crate) fn points_indexed(&self) -> &[usize] {
        &self.indices
    }

    pub(crate) fn point(&self, index: usize) -> &[f64] {
        point(&self.data, self.ndim, index)
    }

    pub(crate) fn bbox(&self, node: usize) -> (&[f64], &[f64]) {
        let start = node * self.ndim;
        (
            &self.bbox_mins[start..start + self.ndim],
            &self.bbox_maxes[start..start + self.ndim],
        )
    }

    fn build_node(&mut self, start: usize, end: usize) -> usize {
        let node_index = self.nodes.len();
        self.nodes.push(Node {
            start,
            end,
            split_dim: 0,
            split_value: 0.0,
            left: None,
            right: None,
            leaf: false,
        });

        let (mins, maxes) = self.compute_bbox(start, end);
        self.bbox_mins.extend_from_slice(&mins);
        self.bbox_maxes.extend_from_slice(&maxes);

        let len = end - start;
        if len <= self.leafsize {
            self.nodes[node_index].leaf = true;
            return node_index;
        }

        let split_dim = widest_dimension(&mins, &maxes);
        let mid = start + len / 2;
        self.indices[start..end].select_nth_unstable_by(mid - start, |lhs, rhs| {
            let lhs_value = self.data[*lhs * self.ndim + split_dim];
            let rhs_value = self.data[*rhs * self.ndim + split_dim];
            lhs_value.total_cmp(&rhs_value)
        });

        let split_value = self.data[self.indices[mid] * self.ndim + split_dim];
        let left = self.build_node(start, mid);
        let right = self.build_node(mid, end);
        self.nodes[node_index] = Node {
            start,
            end,
            split_dim,
            split_value,
            left: Some(left),
            right: Some(right),
            leaf: false,
        };
        node_index
    }

    fn compute_bbox(&self, start: usize, end: usize) -> (Vec<f64>, Vec<f64>) {
        let first = self.point(self.indices[start]);
        let mut mins = first.to_vec();
        let mut maxes = first.to_vec();
        for &point_index in &self.indices[start + 1..end] {
            let coords = self.point(point_index);
            for dim in 0..self.ndim {
                mins[dim] = mins[dim].min(coords[dim]);
                maxes[dim] = maxes[dim].max(coords[dim]);
            }
        }
        (mins, maxes)
    }
}

fn widest_dimension(mins: &[f64], maxes: &[f64]) -> usize {
    let mut best_dim = 0;
    let mut best_span = maxes[0] - mins[0];
    for dim in 1..mins.len() {
        let span = maxes[dim] - mins[dim];
        if span > best_span {
            best_span = span;
            best_dim = dim;
        }
    }
    best_dim
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::array;

    use super::Tree;

    #[test]
    fn build_rejects_empty_inputs() {
        let data = ndarray::Array2::<f64>::zeros((0, 2));
        let result = Tree::new(data.view(), 32);
        assert!(result.is_err());
    }

    #[test]
    fn build_preserves_shape_information() {
        let data = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
        let tree = Tree::new(data.view(), 2).expect("tree should build");

        assert_eq!(tree.n_points(), 4);
        assert_eq!(tree.ndim(), 2);
        assert_eq!(tree.leafsize(), 2);
        let (mins, maxes) = tree.bbox(tree.root());
        assert_relative_eq!(mins[0], 0.0);
        assert_relative_eq!(mins[1], 0.0);
        assert_relative_eq!(maxes[0], 3.0);
        assert_relative_eq!(maxes[1], 3.0);
    }
}
