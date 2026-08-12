/// Flat-array KD-tree node, laid out in preorder (a node's left subtree
/// immediately follows it).
///
/// Inner nodes carry the split plane for O(1) incremental cell bounds during
/// descent; tight per-node bounding boxes stored alongside the node array
/// provide a second, stronger pruning bound that is only consulted when the
/// cheap plane bound fails to prune.
///
/// The top bit of `split_dim` marks nodes whose children shrink
/// substantially along non-split dimensions — the signature of data
/// clustered on a low-dimensional manifold, where the split plane is a poor
/// proxy for actual proximity and queries should order and prune children
/// by tight-box distance instead. The packing is private to this module:
/// build with [`Node::inner`], read with [`unpack_split`].
#[derive(Clone, Copy)]
pub enum Node {
    Leaf {
        start: u32,
        end: u32,
    },
    Inner {
        left: u32,
        right: u32,
        split_dim: u32,
        split_value: f64,
    },
}

/// Flag bit packed into `Node::Inner::split_dim`.
const ORDER_BY_BOX: u32 = 1 << 31;

impl Node {
    pub fn inner(left: u32, right: u32, split_dim: usize, order_by_box: bool, split_value: f64) -> Self {
        debug_assert!(split_dim < ORDER_BY_BOX as usize);
        Node::Inner {
            left,
            right,
            split_dim: split_dim as u32 | if order_by_box { ORDER_BY_BOX } else { 0 },
            split_value,
        }
    }
}

/// Decode a packed `Node::Inner::split_dim` into `(dimension, order_by_box)`.
#[inline(always)]
pub fn unpack_split(split_dim: u32) -> (usize, bool) {
    ((split_dim & !ORDER_BY_BOX) as usize, split_dim & ORDER_BY_BOX != 0)
}
