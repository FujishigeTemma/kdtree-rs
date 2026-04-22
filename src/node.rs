/// Flat-array KD-tree node.
///
/// Inner nodes carry only the split plane (`split_dim`, `split_value`) and the
/// indices of their children. The data-side bounding box is held once at the
/// tree root; descending to a child changes the cell only along `split_dim`,
/// so query-time distance bounds can be updated in O(1).
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
