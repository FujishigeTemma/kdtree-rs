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
            nodes: Vec::with_capacity(2 * n_points.div_ceil(leafsize)),
            root_lo: Vec::new(),
            root_hi: Vec::new(),
            root: 0,
            n_points,
            ndim,
            leafsize,
        };
        let mut scratch_lo = vec![0.0_f64; ndim];
        let mut scratch_hi = vec![0.0_f64; ndim];
        tree.compute_bbox_into(0, n_points, &mut scratch_lo, &mut scratch_hi);
        tree.root_lo = scratch_lo.clone();
        tree.root_hi = scratch_hi.clone();
        let root = tree.build_node(0, n_points, &mut scratch_lo, &mut scratch_hi);
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

    fn build_node(
        &mut self,
        start: usize,
        end: usize,
        scratch_lo: &mut [f64],
        scratch_hi: &mut [f64],
    ) -> u32 {
        let len = end - start;
        if len <= self.leafsize {
            let id = self.nodes.len() as u32;
            self.nodes.push(Node::Leaf {
                start: start as u32,
                end: end as u32,
            });
            return id;
        }

        self.compute_bbox_into(start, end, scratch_lo, scratch_hi);
        let split_dim = widest_dimension(scratch_lo, scratch_hi);
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

        let left = self.build_node(start, mid, scratch_lo, scratch_hi);
        let right = self.build_node(mid, end, scratch_lo, scratch_hi);
        self.nodes[id as usize] = Node::Inner {
            left,
            right,
            split_dim: split_dim as u32,
            split_value,
        };
        id
    }

    fn compute_bbox_into(&self, start: usize, end: usize, lo: &mut [f64], hi: &mut [f64]) {
        let ndim = self.ndim;
        let lo = &mut lo[..ndim];
        let hi = &mut hi[..ndim];
        let first_base = self.indices[start] * ndim;
        let first = &self.data[first_base..first_base + ndim];
        lo.copy_from_slice(first);
        hi.copy_from_slice(first);
        for &point_index in &self.indices[start + 1..end] {
            let base = point_index * ndim;
            let coords = &self.data[base..base + ndim];
            for dim in 0..ndim {
                lo[dim] = lo[dim].min(coords[dim]);
                hi[dim] = hi[dim].max(coords[dim]);
            }
        }
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

    #[test]
    fn build_allocation_count_does_not_scale_with_n() {
        let n = 1000;
        let ndim = 8;
        let leafsize = 16;
        let data: Vec<f64> = (0..n * ndim).map(|i| (i as f64) * 0.001).collect();

        let info = allocation_counter::measure(move || {
            Tree::new(data, n, ndim, leafsize).expect("tree should build");
        });

        assert!(
            info.count_total < 20,
            "expected < 20 allocations, got {}",
            info.count_total
        );
    }
}
