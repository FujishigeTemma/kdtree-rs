//! Tree construction: validation, the preorder layout policy, and the
//! in-place recursive build.
//!
//! # Layout policy
//!
//! [`split_point`] fixes where a subtree of `len` rows splits, which makes
//! the whole preorder layout a pure function of `(n_points, leafsize)`:
//! [`count_nodes`] derives every subtree's node count from it, and
//! [`Subtree::into_children`] uses those counts to carve storage into
//! disjoint child slices. Because the layout never depends on execution
//! order, building children serially or on rayon workers yields identical
//! trees.
//!
//! # Memory discipline
//!
//! The build allocates the four output arrays plus one scratch (`keys`) up
//! front and nothing else: rows are physically partitioned in place, so
//! every recursion level streams contiguous memory and leaves end up
//! contiguous without a separate reorder pass, and the recursion hands each
//! child the disjoint halves of every buffer.

use crate::error::KDTreeError;
use crate::kernel::{bbox, bbox_check_finite};
use crate::tree::{Node, ROOT, Tree, split_box, split_box_mut};

/// Both children of a flagged node must have shrunk below this fraction of
/// the parent's extent in every non-split dimension for the node to be
/// marked `order_by_box` (see [`Node`] docs). Genuinely flat data keeps
/// non-split extents near 1.0; data on a low-dimensional manifold halves
/// them at every split.
const BOX_ORDER_MAX_SHRINK_RATIO: f64 = 0.6;

/// Subtrees smaller than this never get the `order_by_box` flag: their
/// extents are noisy order statistics of few samples (which false-positive
/// on flat data), and ordering barely matters that deep anyway.
const BOX_ORDER_MIN_SUBTREE: usize = 64;

/// Validate the input, then build the tree. The only constructor of [`Tree`].
pub(crate) fn build(
    mut data: Vec<f64>,
    ndim: usize,
    leafsize: usize,
    parallel: bool,
) -> Result<Tree, KDTreeError> {
    if leafsize == 0 {
        return Err(KDTreeError::InvalidLeafsize);
    }
    if ndim == 0 || data.is_empty() {
        return Err(KDTreeError::EmptyData);
    }
    if !data.len().is_multiple_of(ndim) {
        return Err(KDTreeError::InvalidShape(
            "data length must be a multiple of ndim",
        ));
    }
    let n_points = data.len() / ndim;
    if n_points > u32::MAX as usize {
        return Err(KDTreeError::TooManyPoints(n_points));
    }
    let n_nodes = count_nodes(n_points, leafsize);
    if n_nodes > u32::MAX as usize {
        return Err(KDTreeError::TooManyNodes(n_nodes));
    }

    let mut indices = (0..n_points as u32).collect::<Vec<_>>();
    let mut nodes = vec![Node::Leaf { start: 0, end: 0 }; n_nodes];
    let mut bounds = vec![0.0_f64; 2 * ndim * n_nodes];
    let mut keys = vec![0.0_f64; n_points];

    // The root bounding-box pass doubles as the finiteness check, so the
    // whole input is only validated once.
    {
        let (lo, hi) = split_box_mut(&mut bounds, ndim);
        if !bbox_check_finite(&data, ndim, lo, hi) {
            return Err(KDTreeError::NonFiniteData);
        }
    }

    let layout = Layout {
        ndim,
        leafsize,
        parallel,
    };
    let root = Subtree {
        data: &mut data,
        indices: &mut indices,
        nodes: &mut nodes,
        bounds: &mut bounds,
        keys: &mut keys,
        first_row: 0,
        id: ROOT,
    };
    // Monomorphize the hot recursion over the row width so row swaps, key
    // extraction, and the bbox phase merge compile to straight-line code for
    // the common dimensionalities. The dispatch must map each width to the
    // matching instantiation (`0` = dynamic fallback).
    let build_subtree = match ndim {
        1 => build_subtree::<1>,
        2 => build_subtree::<2>,
        3 => build_subtree::<3>,
        4 => build_subtree::<4>,
        8 => build_subtree::<8>,
        16 => build_subtree::<16>,
        _ => build_subtree::<0>,
    };
    // The root enters the box-known entry point directly: the validation
    // pass above already wrote its bounding box.
    build_subtree(&layout, root);

    Ok(Tree {
        data,
        indices,
        nodes,
        bounds,
        n_points,
        ndim,
        leafsize,
    })
}

/// What stays constant across the recursion. `ndim` is every buffer's row
/// stride and `leafsize` the split cutoff — between them they fix the whole
/// layout; `parallel` only chooses how that fixed layout gets filled in.
struct Layout {
    ndim: usize,
    leafsize: usize,
    /// Whether large subtrees may fan out over rayon. Off lets a caller that
    /// is already parallel above us -- N Python threads each building a tree
    /// under free-threaded CPython -- avoid oversubscribing the machine.
    parallel: bool,
}

/// One subtree's disjoint window into every build buffer.
///
/// `data`/`indices`/`keys` cover this subtree's rows, `nodes`/`bounds` its
/// preorder node range (the subtree's root first). `first_row` is the global
/// row offset of `data[0]`, `id` the global id of `nodes[0]`.
struct Subtree<'a> {
    data: &'a mut [f64],
    indices: &'a mut [u32],
    nodes: &'a mut [Node],
    bounds: &'a mut [f64],
    keys: &'a mut [f64],
    first_row: usize,
    id: u32,
}

impl<'a> Subtree<'a> {
    fn len(&self) -> usize {
        self.indices.len()
    }

    /// Reborrow for a recursive call, keeping this handle usable afterwards
    /// (the child bounding boxes feed the parent's `order_by_box` decision).
    fn reborrow(&mut self) -> Subtree<'_> {
        Subtree {
            data: &mut *self.data,
            indices: &mut *self.indices,
            nodes: &mut *self.nodes,
            bounds: &mut *self.bounds,
            keys: &mut *self.keys,
            first_row: self.first_row,
            id: self.id,
        }
    }

    /// Carve this subtree's storage into the root's own slots plus the two
    /// children's disjoint windows: `mid` rows go left, and per the preorder
    /// layout the left child owns the `left_nodes` node slots directly after
    /// the root. Returns `(own node, own box, left, right)`.
    fn into_children(
        self,
        ndim: usize,
        mid: usize,
        left_nodes: usize,
    ) -> (&'a mut Node, &'a [f64], Subtree<'a>, Subtree<'a>) {
        let (left_data, right_data) = self.data.split_at_mut(mid * ndim);
        let (left_indices, right_indices) = self.indices.split_at_mut(mid);
        let (node, child_nodes) = self
            .nodes
            .split_first_mut()
            .expect("subtree has a root node");
        let (left_node_slots, right_node_slots) = child_nodes.split_at_mut(left_nodes);
        let (own_bounds, child_bounds) = self.bounds.split_at_mut(2 * ndim);
        let (left_bounds, right_bounds) = child_bounds.split_at_mut(2 * ndim * left_nodes);
        let (left_keys, right_keys) = self.keys.split_at_mut(mid);
        let left = Subtree {
            data: left_data,
            indices: left_indices,
            nodes: left_node_slots,
            bounds: left_bounds,
            keys: left_keys,
            first_row: self.first_row,
            id: self.id + 1,
        };
        let right = Subtree {
            data: right_data,
            indices: right_indices,
            nodes: right_node_slots,
            bounds: right_bounds,
            keys: right_keys,
            first_row: self.first_row + mid,
            id: self.id + 1 + left_nodes as u32,
        };
        (node, own_bounds, left, right)
    }
}

/// The split position of a node over `len` points; the anchor of the whole
/// layout policy (see module docs).
#[inline]
fn split_point(len: usize) -> usize {
    len / 2
}

/// Number of nodes in the subtree over `len` points; fully determined by
/// `len` alone because the split position is a function of `len`.
fn count_nodes(len: usize, leafsize: usize) -> usize {
    if len <= leafsize {
        1
    } else {
        let mid = split_point(len);
        1 + count_nodes(mid, leafsize) + count_nodes(len - mid, leafsize)
    }
}

/// Compute `st`'s bounding box, then build it. Every node but the root
/// enters here; `D` is the compile-time row width (`0` = use `l.ndim`
/// dynamically).
fn bound_and_build<const D: usize>(l: &Layout, st: Subtree<'_>) {
    let ndim = if D > 0 { D } else { l.ndim };
    let (lo, hi) = split_box_mut(st.bounds, ndim);
    bbox::<D>(st.data, ndim, lo, hi);
    build_subtree::<D>(l, st);
}

/// Recursively build one subtree in place, given that its own bounding box
/// is already written to `st.bounds`.
fn build_subtree<const D: usize>(l: &Layout, st: Subtree<'_>) {
    debug_assert!(D == 0 || D == l.ndim);
    let ndim = if D > 0 { D } else { l.ndim };
    let len = st.len();
    debug_assert_eq!(st.nodes.len(), count_nodes(len, l.leafsize));

    if len <= l.leafsize {
        st.nodes[0] = Node::Leaf {
            start: st.first_row as u32,
            end: (st.first_row + len) as u32,
        };
        return;
    }

    let split_dim = {
        let (lo, hi) = split_box(st.bounds, ndim);
        widest_dimension(lo, hi)
    };
    let mid = split_point(len);

    // Median select runs on a contiguous copy of the split coordinates, then
    // a three-way partition moves whole rows once. This avoids the gather
    // per comparison that dominates an index-indirected select.
    for (key, row) in st.keys[..len].iter_mut().zip(st.data.chunks_exact(ndim)) {
        *key = row[split_dim];
    }
    let (_, pivot, _) = st.keys[..len].select_nth_unstable_by(mid, f64::total_cmp);
    let pivot = *pivot;
    partition_rows::<D>(st.data, st.indices, ndim, split_dim, pivot, mid);

    let left_nodes = count_nodes(mid, l.leafsize);
    let (node, own_box, mut left, mut right) = st.into_children(ndim, mid, left_nodes);
    let (left_id, right_id) = (left.id, right.id);

    if l.parallel {
        rayon::join(
            || bound_and_build::<D>(l, left.reborrow()),
            || bound_and_build::<D>(l, right.reborrow()),
        );
    } else {
        bound_and_build::<D>(l, left.reborrow());
        bound_and_build::<D>(l, right.reborrow());
    }

    // With both children's tight boxes known, decide whether queries should
    // order this node's children by box distance during their initial
    // descent (see `Node` docs). Shrinkage along non-split dimensions means
    // the data lives on a lower-dimensional manifold and the split plane
    // alone misjudges proximity.
    let order_by_box = ndim > 1
        && len >= BOX_ORDER_MIN_SUBTREE
        && shrank_off_axis(own_box, left.bounds, ndim, split_dim)
        && shrank_off_axis(own_box, right.bounds, ndim, split_dim);
    *node = Node::Inner {
        left: left_id,
        right: right_id,
        split_dim: split_dim as u32,
        order_by_box,
        split_value: pivot,
    };
}

/// Did `child`'s extent shrink below [`BOX_ORDER_MAX_SHRINK_RATIO`] of
/// `parent`'s in every dimension other than the split?
fn shrank_off_axis(parent: &[f64], child: &[f64], ndim: usize, split_dim: usize) -> bool {
    let (parent_lo, parent_hi) = split_box(parent, ndim);
    let (child_lo, child_hi) = split_box(child, ndim);
    let mut max_ratio = 0.0_f64;
    for d in 0..ndim {
        if d == split_dim {
            continue;
        }
        let parent_extent = parent_hi[d] - parent_lo[d];
        let ratio = if parent_extent > 0.0 {
            (child_hi[d] - child_lo[d]) / parent_extent
        } else {
            1.0
        };
        max_ratio = max_ratio.max(ratio);
    }
    max_ratio < BOX_ORDER_MAX_SHRINK_RATIO
}

/// Partition whole rows around `pivot` on `split_dim` so that rows before
/// `mid` have keys `<= pivot` and rows from `mid` on have keys `>= pivot`,
/// no matter how many duplicates exist.
///
/// A Hoare-style two-pointer pass on `< pivot` swaps only misplaced pairs
/// (a Dutch-flag loop moves ~3x as many rows), then a fix-up walk pulls
/// pivot-equal rows into the `[lt, mid)` gap — a no-op when the pivot is
/// unique and already adjacent, the common case.
fn partition_rows<const D: usize>(
    data: &mut [f64],
    indices: &mut [u32],
    ndim: usize,
    split_dim: usize,
    pivot: f64,
    mid: usize,
) {
    debug_assert_eq!(data.len(), indices.len() * ndim);
    debug_assert!(D == 0 || D == ndim);
    let ndim = if D > 0 { D } else { ndim };
    let len = indices.len();
    let mut i = 0;
    let mut j = len;
    loop {
        while i < j && data[i * ndim + split_dim] < pivot {
            i += 1;
        }
        while i < j && data[(j - 1) * ndim + split_dim] >= pivot {
            j -= 1;
        }
        if i >= j {
            break;
        }
        j -= 1;
        swap_rows::<D>(data, ndim, i, j);
        indices.swap(i, j);
        i += 1;
    }
    // `[0, i)` holds every `< pivot` row; the mid-th order statistic being
    // `pivot` guarantees enough pivot-equal rows in the tail to fill up to
    // `mid`.
    let mut place = i;
    let mut scan = i;
    while place < mid {
        debug_assert!(scan < len);
        if data[scan * ndim + split_dim] == pivot {
            swap_rows::<D>(data, ndim, scan, place);
            indices.swap(scan, place);
            place += 1;
        }
        scan += 1;
    }
}

/// Swap two full rows. With a compile-time width both the row addressing and
/// the copy fully unroll; the dynamic fallback is a single
/// `swap_nonoverlapping` of `ndim` elements.
#[inline(always)]
fn swap_rows<const D: usize>(data: &mut [f64], ndim: usize, a: usize, b: usize) {
    if a == b {
        return;
    }
    let ndim = if D > 0 { D } else { ndim };
    debug_assert!(a * ndim + ndim <= data.len() && b * ndim + ndim <= data.len());
    // SAFETY: `a != b` guarantees the two `ndim`-element rows are disjoint,
    // and both row ranges are in bounds per the debug assertion above
    // (callers index rows within `data.len() / ndim`).
    unsafe {
        let base = data.as_mut_ptr();
        std::ptr::swap_nonoverlapping(base.add(a * ndim), base.add(b * ndim), ndim);
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
    use crate::tree::Tree;

    #[test]
    fn build_rejects_empty_inputs() {
        let result = Tree::new(Vec::new(), 2, 32, true);
        assert!(result.is_err());
    }

    #[test]
    fn build_allocation_count_does_not_scale_with_n() {
        let n = 1000;
        let ndim = 8;
        let leafsize = 16;
        let data: Vec<f64> = (0..n * ndim).map(|i| (i as f64) * 0.001).collect();

        let info = allocation_counter::measure(move || {
            Tree::new(data, ndim, leafsize, true).expect("tree should build");
        });

        assert!(
            info.count_total < 20,
            "expected < 20 allocations, got {}",
            info.count_total
        );
    }
}
