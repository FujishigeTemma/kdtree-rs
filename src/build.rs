//! [`split_point`] is the anchor of the whole design: fixing the split at
//! `len / 2` makes the preorder layout a pure function of `(n_points,
//! leafsize)`. [`count_nodes`] can then derive every subtree's node count in
//! advance, [`Subtree::into_children`] can carve disjoint `&mut` windows out of
//! every buffer, and the two children can be built on rayon workers with no
//! locking and no merge pass — producing a tree identical to the serial one.
//! Change the split policy and none of that holds.

use crate::error::KDTreeError;
use crate::layout::{BBox, BBoxMut, Boxes, BoxesMut, Dyn, Rows, RowsMut, Width, with_width};
use crate::tree::{Node, ROOT, Tree};

/// Every non-split dimension of both children must be below this fraction of
/// the parent's extent for `order_by_box`. Flat data keeps non-split extents
/// near 1.0; data on a low-dimensional manifold halves them at every split.
const BOX_ORDER_MAX_SHRINK_RATIO: f64 = 0.6;

/// Below this size the extents are noisy order statistics of few samples, which
/// false-positive on flat data.
const BOX_ORDER_MIN_SUBTREE: usize = 64;

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
    let mut boxes = Boxes::zeroed(n_nodes, ndim);
    let mut keys = vec![0.0_f64; n_points];

    {
        let mut all_boxes = boxes.as_mut();
        let mut root_box = all_boxes.first_mut();
        if !Rows::new(&data, Dyn(ndim)).bbox_checked_into(&mut root_box) {
            return Err(KDTreeError::NonFiniteData);
        }
    }

    let layout = Layout { leafsize, parallel };
    // The root skips `bound_and_build`: the validation pass above already wrote
    // its box. The listed widths are the ones whose row swaps, key extraction,
    // and bbox phase merge are worth unrolling.
    with_width!(ndim, [1, 2, 3, 4, 8, 16], |w| build_subtree(
        &layout,
        Subtree {
            rows: RowsMut::new(&mut data, w),
            indices: &mut indices,
            nodes: &mut nodes,
            boxes: boxes.as_mut(),
            keys: &mut keys,
            first_row: 0,
            id: ROOT,
        }
    ));

    Ok(Tree {
        data,
        indices,
        nodes,
        boxes,
        n_points,
        ndim,
        leafsize,
    })
}

struct Layout {
    leafsize: usize,
    parallel: bool,
}

/// One subtree's disjoint window into every build buffer. `first_row` is the
/// global row offset of `rows[0]`, `id` the global id of `nodes[0]`.
struct Subtree<'a, W: Width> {
    rows: RowsMut<'a, W>,
    indices: &'a mut [u32],
    nodes: &'a mut [Node],
    boxes: BoxesMut<'a>,
    keys: &'a mut [f64],
    first_row: usize,
    id: u32,
}

impl<'a, W: Width> Subtree<'a, W> {
    fn len(&self) -> usize {
        self.indices.len()
    }

    /// Keeps this handle usable after the recursive call, which is what lets the
    /// parent read the child boxes for its `order_by_box` decision.
    fn reborrow(&mut self) -> Subtree<'_, W> {
        Subtree {
            rows: self.rows.reborrow(),
            indices: &mut *self.indices,
            nodes: &mut *self.nodes,
            boxes: self.boxes.reborrow(),
            keys: &mut *self.keys,
            first_row: self.first_row,
            id: self.id,
        }
    }

    fn into_children(
        self,
        mid: usize,
        left_nodes: usize,
    ) -> (&'a mut Node, BBoxMut<'a>, Subtree<'a, W>, Subtree<'a, W>) {
        let (left_rows, right_rows) = self.rows.split_at(mid);
        let (left_indices, right_indices) = self.indices.split_at_mut(mid);
        let (node, child_nodes) = self
            .nodes
            .split_first_mut()
            .expect("subtree has a root node");
        let (left_node_slots, right_node_slots) = child_nodes.split_at_mut(left_nodes);
        let (own_box, child_boxes) = self.boxes.split_first();
        let (left_boxes, right_boxes) = child_boxes.split_at(left_nodes);
        let (left_keys, right_keys) = self.keys.split_at_mut(mid);
        let left = Subtree {
            rows: left_rows,
            indices: left_indices,
            nodes: left_node_slots,
            boxes: left_boxes,
            keys: left_keys,
            first_row: self.first_row,
            id: self.id + 1,
        };
        let right = Subtree {
            rows: right_rows,
            indices: right_indices,
            nodes: right_node_slots,
            boxes: right_boxes,
            keys: right_keys,
            first_row: self.first_row + mid,
            id: self.id + 1 + left_nodes as u32,
        };
        (node, own_box, left, right)
    }
}

#[inline]
fn split_point(len: usize) -> usize {
    len / 2
}

fn count_nodes(len: usize, leafsize: usize) -> usize {
    if len <= leafsize {
        1
    } else {
        let mid = split_point(len);
        1 + count_nodes(mid, leafsize) + count_nodes(len - mid, leafsize)
    }
}

fn bound_and_build<W: Width>(l: &Layout, mut st: Subtree<'_, W>) {
    {
        let mut own = st.boxes.first_mut();
        st.rows.as_ref().bbox_into(&mut own);
    }
    build_subtree(l, st);
}

/// Requires `st`'s own box to be written already.
fn build_subtree<W: Width>(l: &Layout, mut st: Subtree<'_, W>) {
    let len = st.len();
    debug_assert_eq!(st.nodes.len(), count_nodes(len, l.leafsize));

    if len <= l.leafsize {
        st.nodes[0] = Node::Leaf {
            start: st.first_row as u32,
            end: (st.first_row + len) as u32,
        };
        return;
    }

    let split_dim = st.boxes.first().widest_axis();
    let mid = split_point(len);

    // Selecting on a contiguous copy of the split coordinates avoids the gather
    // per comparison that dominates an index-indirected select.
    for (key, row) in st.keys[..len].iter_mut().zip(st.rows.as_ref().iter()) {
        *key = row[split_dim];
    }
    let (_, pivot, _) = st.keys[..len].select_nth_unstable_by(mid, f64::total_cmp);
    let pivot = *pivot;
    partition_rows(st.rows.reborrow(), st.indices, split_dim, pivot, mid);

    let left_nodes = count_nodes(mid, l.leafsize);
    let (node, own_box, mut left, mut right) = st.into_children(mid, left_nodes);
    let (left_id, right_id) = (left.id, right.id);

    if l.parallel {
        rayon::join(
            || bound_and_build(l, left.reborrow()),
            || bound_and_build(l, right.reborrow()),
        );
    } else {
        bound_and_build(l, left.reborrow());
        bound_and_build(l, right.reborrow());
    }

    let own_box = own_box.as_ref();
    let order_by_box = own_box.ndim() > 1
        && len >= BOX_ORDER_MIN_SUBTREE
        && shrank_off_axis(own_box, left.boxes.first(), split_dim)
        && shrank_off_axis(own_box, right.boxes.first(), split_dim);
    *node = Node::Inner {
        left: left_id,
        right: right_id,
        split_dim: split_dim as u32,
        order_by_box,
        split_value: pivot,
    };
}

fn shrank_off_axis(parent: BBox<'_>, child: BBox<'_>, split_dim: usize) -> bool {
    let mut max_ratio = 0.0_f64;
    for d in 0..parent.ndim() {
        if d == split_dim {
            continue;
        }
        let parent_extent = parent.extent(d);
        let ratio = if parent_extent > 0.0 {
            child.extent(d) / parent_extent
        } else {
            1.0
        };
        max_ratio = max_ratio.max(ratio);
    }
    max_ratio < BOX_ORDER_MAX_SHRINK_RATIO
}

/// Move rows so everything before `mid` has key `<= pivot` and everything from
/// `mid` on has key `>= pivot`, however many duplicates exist.
///
/// The Hoare pass swaps only misplaced pairs; a Dutch-flag loop moves about
/// three times as many rows. The fix-up walk then pulls pivot-equal rows into
/// the `[lt, mid)` gap, and is a no-op when the pivot is unique.
///
/// `rows` is taken by value, not `&mut`: behind a reference the row width stays
/// in memory and the two-pointer loop reloads it every step (12.5% on a d5
/// build).
fn partition_rows<W: Width>(
    mut rows: RowsMut<'_, W>,
    indices: &mut [u32],
    split_dim: usize,
    pivot: f64,
    mid: usize,
) {
    debug_assert_eq!(rows.len(), indices.len());
    let len = indices.len();
    let mut i = 0;
    let mut j = len;
    loop {
        while i < j && rows.coord(i, split_dim) < pivot {
            i += 1;
        }
        while i < j && rows.coord(j - 1, split_dim) >= pivot {
            j -= 1;
        }
        if i >= j {
            break;
        }
        j -= 1;
        rows.swap(i, j);
        indices.swap(i, j);
        i += 1;
    }
    // `[0, i)` holds every `< pivot` row; the mid-th order statistic being
    // `pivot` guarantees enough pivot-equal rows in the tail to reach `mid`.
    let mut place = i;
    let mut scan = i;
    while place < mid {
        debug_assert!(scan < len);
        if rows.coord(scan, split_dim) == pivot {
            rows.swap(scan, place);
            indices.swap(scan, place);
            place += 1;
        }
        scan += 1;
    }
}

#[cfg(test)]
mod tests {
    use crate::tree::{Node, Tree};

    #[test]
    fn build_rejects_empty_inputs() {
        assert!(Tree::new(Vec::new(), 2, 32, true).is_err());
    }

    /// Serial on purpose: `allocation_counter` is a global hook, so a parallel
    /// build also counts rayon's pool and work-deque allocations — a number that
    /// depends on how much stealing happened, not on anything this crate does.
    #[test]
    fn build_allocation_count_does_not_scale_with_n() {
        let n = 1000;
        let ndim = 8;
        let leafsize = 16;
        let data: Vec<f64> = (0..n * ndim).map(|i| (i as f64) * 0.001).collect();

        let info = allocation_counter::measure(move || {
            Tree::new(data, ndim, leafsize, false).expect("tree should build");
        });

        assert!(
            info.count_total < 20,
            "expected < 20 allocations, got {}",
            info.count_total
        );
    }

    /// The layout is a pure function of `(n_points, leafsize)` (see module
    /// docs), so rayon has to produce a bit-identical tree.
    #[test]
    fn parallel_build_matches_serial_build() {
        let (n, ndim, leafsize) = (2_000, 5, 8);
        let data: Vec<f64> = (0..n * ndim)
            .map(|i| ((i * 2_654_435_761_usize) % 10_007) as f64)
            .collect();

        let serial = Tree::new(data.clone(), ndim, leafsize, false).expect("serial build");
        let parallel = Tree::new(data, ndim, leafsize, true).expect("parallel build");

        assert_eq!(serial.data, parallel.data);
        assert_eq!(serial.indices, parallel.indices);
        assert_eq!(serial.nodes.len(), parallel.nodes.len());
        for id in 0..serial.nodes.len() as u32 {
            let (a, b) = (serial.box_of(id), parallel.box_of(id));
            assert_eq!(a.lo, b.lo);
            assert_eq!(a.hi, b.hi);
            match (serial.node(id), parallel.node(id)) {
                (
                    Node::Leaf { start: s0, end: e0 },
                    Node::Leaf {
                        start: s1,
                        end: e1,
                    },
                ) => assert_eq!((s0, e0), (s1, e1)),
                (
                    Node::Inner {
                        left: l0,
                        right: r0,
                        split_dim: d0,
                        order_by_box: o0,
                        split_value: v0,
                    },
                    Node::Inner {
                        left: l1,
                        right: r1,
                        split_dim: d1,
                        order_by_box: o1,
                        split_value: v1,
                    },
                ) => assert_eq!((l0, r0, d0, o0, v0), (l1, r1, d1, o1, v1)),
                _ => panic!("node {id} differs in kind"),
            }
        }
    }
}
