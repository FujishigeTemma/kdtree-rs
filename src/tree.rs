use crate::error::KDTreeError;
use crate::node::Node;

pub struct Tree {
    data: Vec<f64>,
    indices: Vec<usize>,
    nodes: Vec<Node>,
    root_lo: Vec<f64>,
    root_hi: Vec<f64>,
    root: u32,
    n_points: usize,
    ndim: usize,
    leafsize: usize,
}

impl Tree {
    /// Build a tree from a row-major `Vec<f64>` of length `n_points * ndim`.
    /// Takes the data by value so the caller can release any Python borrow
    /// before invoking us and we can run under `py.detach`.
    pub fn new(
        data: Vec<f64>,
        n_points: usize,
        ndim: usize,
        leafsize: usize,
    ) -> Result<Self, KDTreeError> {
        if leafsize == 0 {
            return Err(KDTreeError::InvalidLeafsize);
        }
        if n_points == 0 || ndim == 0 {
            return Err(KDTreeError::EmptyData);
        }
        if data.len() != n_points * ndim {
            return Err(KDTreeError::InvalidShape(
                "data length must equal n_points * ndim",
            ));
        }
        if !data.iter().all(|value| value.is_finite()) {
            return Err(KDTreeError::NonFiniteData);
        }

        let indices = (0..n_points).collect::<Vec<_>>();

        let mut tree = Self {
            data,
            indices,
            nodes: Vec::with_capacity(2 * n_points.div_ceil(leafsize.max(1))),
            root_lo: Vec::new(),
            root_hi: Vec::new(),
            root: 0,
            n_points,
            ndim,
            leafsize,
        };
        let (lo, hi) = tree.compute_bbox(0, n_points);
        tree.root_lo = lo;
        tree.root_hi = hi;
        let root = tree.build_node(0, n_points);
        tree.root = root;
        tree.reorder_leaves_contiguous();
        Ok(tree)
    }

    /// Reconstruct the original-order data the caller passed to `new`.
    /// Internally we keep the leaf-reordered layout for query cache locality;
    /// the original order is only materialized on demand for the `data` getter.
    pub fn original_data(&self) -> Vec<f64> {
        let mut original = vec![0.0_f64; self.n_points * self.ndim];
        for (pos, &original_idx) in self.indices.iter().enumerate() {
            let src = pos * self.ndim;
            let dst = original_idx * self.ndim;
            original[dst..dst + self.ndim].copy_from_slice(&self.data[src..src + self.ndim]);
        }
        original
    }

    /// Permute `data` so that points within each leaf live contiguously in
    /// tree-position order. After this, `self.data[pos * ndim ..]` is the
    /// point at tree position `pos`, and `self.indices[pos]` is that point's
    /// original data index. Subsequent queries iterate leaves sequentially.
    fn reorder_leaves_contiguous(&mut self) {
        let mut reordered = Vec::with_capacity(self.data.len());
        for &original in &self.indices {
            let start = original * self.ndim;
            reordered.extend_from_slice(&self.data[start..start + self.ndim]);
        }
        self.data = reordered;
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

    pub(crate) fn root(&self) -> u32 {
        self.root
    }

    pub(crate) fn node(&self, index: u32) -> &Node {
        &self.nodes[index as usize]
    }

    pub(crate) fn points_indexed(&self) -> &[usize] {
        &self.indices
    }

    pub(crate) fn root_bbox(&self) -> (&[f64], &[f64]) {
        (&self.root_lo, &self.root_hi)
    }

    /// Return the contiguous slice of coordinates for tree positions
    /// `[start, end)`. After the leaf reorder this corresponds exactly to the
    /// points in the leaf/subtree, laid out row-major.
    pub(crate) fn leaf_block(&self, start: usize, end: usize) -> &[f64] {
        &self.data[start * self.ndim..end * self.ndim]
    }

    fn build_node(&mut self, start: usize, end: usize) -> u32 {
        let len = end - start;
        if len <= self.leafsize {
            let id = self.nodes.len() as u32;
            self.nodes.push(Node::Leaf {
                start: start as u32,
                end: end as u32,
            });
            return id;
        }

        let (mins, maxes) = self.compute_bbox(start, end);
        let split_dim = widest_dimension(&mins, &maxes);
        let mid = start + len / 2;
        let ndim = self.ndim;
        let data = &self.data;
        self.indices[start..end].select_nth_unstable_by(mid - start, |lhs, rhs| {
            let lhs_value = data[*lhs * ndim + split_dim];
            let rhs_value = data[*rhs * ndim + split_dim];
            lhs_value.total_cmp(&rhs_value)
        });
        let split_value = self.data[self.indices[mid] * ndim + split_dim];

        let id = self.nodes.len() as u32;
        self.nodes.push(Node::Leaf { start: 0, end: 0 }); // placeholder

        let left = self.build_node(start, mid);
        let right = self.build_node(mid, end);
        self.nodes[id as usize] = Node::Inner {
            left,
            right,
            split_dim: split_dim as u32,
            split_value,
        };
        id
    }

    fn compute_bbox(&self, start: usize, end: usize) -> (Vec<f64>, Vec<f64>) {
        let ndim = self.ndim;
        let row = |idx: usize| {
            let base = idx * ndim;
            &self.data[base..base + ndim]
        };
        let first = row(self.indices[start]);
        let mut mins = first.to_vec();
        let mut maxes = first.to_vec();
        for &point_index in &self.indices[start + 1..end] {
            let coords = row(point_index);
            for dim in 0..ndim {
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

    use super::Tree;

    #[test]
    fn build_rejects_empty_inputs() {
        let result = Tree::new(Vec::new(), 0, 2, 32);
        assert!(result.is_err());
    }

    #[test]
    fn build_preserves_shape_information() {
        let data = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
        let tree = Tree::new(data, 4, 2, 2).expect("tree should build");

        assert_eq!(tree.n_points(), 4);
        assert_eq!(tree.ndim(), 2);
        assert_eq!(tree.leafsize(), 2);
        let (mins, maxes) = tree.root_bbox();
        assert_relative_eq!(mins[0], 0.0);
        assert_relative_eq!(mins[1], 0.0);
        assert_relative_eq!(maxes[0], 3.0);
        assert_relative_eq!(maxes[1], 3.0);
    }

    #[test]
    fn original_data_round_trips() {
        let data = vec![5.0, 1.0, 2.0, 4.0, 0.0, 3.0, 6.0, 7.0];
        let tree = Tree::new(data.clone(), 4, 2, 1).expect("tree should build");
        assert_eq!(tree.original_data(), data);
    }
}
