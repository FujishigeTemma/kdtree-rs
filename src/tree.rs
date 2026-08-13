//! The tree itself: storage layout and read access. Construction lives in
//! `build.rs` (the only module that writes these fields), queries in
//! `query.rs`.
//!
//! # Storage layout
//!
//! Nodes are laid out flat in *preorder*: a node's left subtree immediately
//! follows it, then its right subtree. Points are physically reordered
//! during the build so that every subtree — and in particular every leaf —
//! owns one contiguous, row-major block of `data`; `indices[pos]` maps a
//! tree position back to the caller's original row. Per-node tight bounding
//! boxes live in a parallel flat array, `2 * ndim` values per node.

use crate::error::KDTreeError;

/// Id of the root node: the build lays nodes out in preorder starting here.
pub(crate) const ROOT: u32 = 0;

/// Split one node's `2 * ndim` bounds slot into its `(lo, hi)` halves. The
/// build writes these slots and queries read them, so the `[lo | hi]` layout
/// is spelled out here once rather than at every call site.
#[inline]
pub(crate) fn split_box(bounds: &[f64], ndim: usize) -> (&[f64], &[f64]) {
    bounds[..2 * ndim].split_at(ndim)
}

/// [`split_box`] for the build's write side.
#[inline]
pub(crate) fn split_box_mut(bounds: &mut [f64], ndim: usize) -> (&mut [f64], &mut [f64]) {
    bounds[..2 * ndim].split_at_mut(ndim)
}

/// One flat-array node.
///
/// Inner nodes carry the split plane for O(1) incremental cell bounds during
/// descent; the tight per-node bounding boxes stored alongside the node
/// array provide a second, stronger pruning bound that is only consulted
/// when the cheap plane bound fails to prune.
///
/// `order_by_box` marks nodes whose children shrink substantially along
/// non-split dimensions — the signature of data clustered on a
/// low-dimensional manifold, where the split plane is a poor proxy for
/// actual proximity and queries should order and prune children by tight-box
/// distance instead.
#[derive(Clone, Copy)]
pub(crate) enum Node {
    Leaf {
        start: u32,
        end: u32,
    },
    Inner {
        left: u32,
        right: u32,
        split_dim: u32,
        order_by_box: bool,
        split_value: f64,
    },
}

pub struct Tree {
    /// Point coordinates in tree order (see module docs), row-major.
    pub(crate) data: Vec<f64>,
    /// Tree position -> original row index.
    pub(crate) indices: Vec<u32>,
    /// Preorder node array.
    pub(crate) nodes: Vec<Node>,
    /// Tight per-node bounding boxes, `2 * ndim` values per node laid out as
    /// `[lo[0..ndim], hi[0..ndim]]`. Queries prune against these, which
    /// bounds the distance to a subtree by the points it actually contains
    /// rather than by the (much looser) space partition of split planes.
    pub(crate) bounds: Vec<f64>,
    pub(crate) n_points: usize,
    pub(crate) ndim: usize,
    pub(crate) leafsize: usize,
}

impl Tree {
    /// Build a tree from a row-major `Vec<f64>` of `ndim`-wide points.
    /// Takes the data by value so the caller can release any Python borrow
    /// before invoking us and we can run under `py.detach`.
    pub fn new(
        data: Vec<f64>,
        ndim: usize,
        leafsize: usize,
        parallel: bool,
    ) -> Result<Self, KDTreeError> {
        crate::build::build(data, ndim, leafsize, parallel)
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

    /// Reconstruct the original-order data the caller passed to `new`.
    /// Internally we keep the tree-ordered layout for query cache locality;
    /// the original order is only materialized on demand for the `data`
    /// getter.
    pub fn original_data(&self) -> Vec<f64> {
        let mut original = vec![0.0_f64; self.n_points * self.ndim];
        for (pos, &original_idx) in self.indices.iter().enumerate() {
            let src = pos * self.ndim;
            let dst = original_idx as usize * self.ndim;
            original[dst..dst + self.ndim].copy_from_slice(&self.data[src..src + self.ndim]);
        }
        original
    }

    pub(crate) fn node(&self, index: u32) -> &Node {
        &self.nodes[index as usize]
    }

    /// Bounding box of node `index` as `(lo, hi)` slices of length `ndim`.
    pub(crate) fn box_of(&self, index: u32) -> (&[f64], &[f64]) {
        let base = 2 * self.ndim * index as usize;
        split_box(&self.bounds[base..], self.ndim)
    }

    pub(crate) fn root_box(&self) -> (&[f64], &[f64]) {
        self.box_of(ROOT)
    }

    /// The contiguous slice of coordinates for tree positions `[start, end)`
    /// — exactly the points of that leaf/subtree, row-major.
    pub(crate) fn rows(&self, start: usize, end: usize) -> &[f64] {
        &self.data[start * self.ndim..end * self.ndim]
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::{Node, Tree};

    #[test]
    fn node_stays_within_one_cache_slot() {
        // `Inner`'s three `u32`s plus the `bool` fit in the padding the
        // `f64`'s alignment already requires, so `order_by_box` costs no
        // space; this guards that the layout stays that way.
        assert!(size_of::<Node>() <= 24);
    }

    #[test]
    fn build_preserves_shape_information() {
        let data = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
        let tree = Tree::new(data, 2, 2, true).expect("tree should build");

        assert_eq!(tree.n_points(), 4);
        assert_eq!(tree.ndim(), 2);
        assert_eq!(tree.leafsize(), 2);
        let (mins, maxes) = tree.root_box();
        assert_relative_eq!(mins[0], 0.0);
        assert_relative_eq!(mins[1], 0.0);
        assert_relative_eq!(maxes[0], 3.0);
        assert_relative_eq!(maxes[1], 3.0);
    }

    #[test]
    fn original_data_round_trips() {
        let data = vec![5.0, 1.0, 2.0, 4.0, 0.0, 3.0, 6.0, 7.0];
        let tree = Tree::new(data.clone(), 2, 1, true).expect("tree should build");
        assert_eq!(tree.original_data(), data);
    }
}
