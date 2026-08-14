//! Nodes are laid out flat in preorder: a node's left subtree immediately
//! follows it, then its right subtree. Points are physically reordered during
//! the build so every subtree — in particular every leaf — owns one contiguous
//! row-major block of `data`, and `indices[position]` maps back to the caller's
//! original row.

use crate::error::KDTreeError;
use crate::layout::{BBox, Boxes, Dyn, Rows};

pub(crate) const ROOT: u32 = 0;

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
        /// Both children shrank along every non-split dimension, which marks
        /// data on a low-dimensional manifold: there the split plane is a poor
        /// proxy for proximity, so queries order and prune this node's children
        /// by tight-box distance instead.
        order_by_box: bool,
        split_value: f64,
    },
}

pub struct Tree {
    /// Row-major, in tree order (see module docs).
    pub(crate) data: Vec<f64>,
    /// Tree position -> original row index.
    pub(crate) indices: Vec<u32>,
    pub(crate) nodes: Vec<Node>,
    /// Tight per-node boxes, which bound the distance to a subtree by the points
    /// it actually contains rather than by the split planes around it.
    pub(crate) boxes: Boxes,
    pub(crate) n_points: usize,
    pub(crate) ndim: usize,
    pub(crate) leafsize: usize,
}

impl Tree {
    /// Takes the data by value so the caller can drop any Python borrow before
    /// calling and run this detached from the GIL.
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

    /// Only the `data` getter needs this; queries keep the tree order for
    /// locality.
    pub fn original_data(&self) -> Vec<f64> {
        let mut original = vec![0.0_f64; self.n_points * self.ndim];
        for (pos, &original_idx) in self.indices.iter().enumerate() {
            let src = pos * self.ndim;
            let dst = original_idx as usize * self.ndim;
            original[dst..dst + self.ndim].copy_from_slice(&self.data[src..src + self.ndim]);
        }
        original
    }

    #[inline(always)]
    pub(crate) fn node(&self, id: u32) -> &Node {
        &self.nodes[id as usize]
    }

    #[inline(always)]
    pub(crate) fn box_of(&self, id: u32) -> BBox<'_> {
        self.boxes.of(id)
    }

    #[inline(always)]
    pub(crate) fn root_box(&self) -> BBox<'_> {
        self.boxes.of(ROOT)
    }

    #[inline(always)]
    pub(crate) fn rows(&self, start: usize, end: usize) -> Rows<'_, Dyn> {
        Rows::new(&self.data, Dyn(self.ndim)).slice(start, end)
    }
}

#[cfg(test)]
mod tests {
    use super::Node;

    /// `Inner`'s `u32`s and `bool` have to keep fitting in the padding the
    /// `f64`'s alignment already requires.
    #[test]
    fn node_stays_within_one_cache_slot() {
        assert!(size_of::<Node>() <= 24);
    }
}
