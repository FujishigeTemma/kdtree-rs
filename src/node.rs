/// Flat-array KD-tree node, laid out in preorder (a node's left subtree
/// immediately follows it).
///
/// Inner nodes carry the split plane for O(1) incremental cell bounds during
/// descent; tight per-node bounding boxes stored alongside the node array
/// provide a second, stronger pruning bound that is only consulted when the
/// cheap plane bound fails to prune.
///
/// The top bit of `split_dim` (`ORDER_BY_BOX`) marks nodes whose children
/// shrink substantially along non-split dimensions — the signature of data
/// clustered on a low-dimensional manifold, where the split plane is a poor
/// proxy for actual proximity and the initial descent should order children
/// by tight-box distance instead.
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
pub const ORDER_BY_BOX: u32 = 1 << 31;
