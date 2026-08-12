use std::simd::prelude::*;

use crate::error::KDTreeError;
use crate::node::Node;
use crate::simd::{F64s, LANES, nonfinite_lanes, vmax, vmin};

/// Subtrees at or above this many points are built on rayon worker threads;
/// smaller ones stay on the current thread so the fork/join overhead never
/// dominates and small builds remain allocation-free.
const PARALLEL_BUILD_THRESHOLD: usize = 1024;

pub struct Tree {
    data: Vec<f64>,
    indices: Vec<u32>,
    nodes: Vec<Node>,
    /// Tight per-node bounding boxes, `2 * ndim` values per node laid out as
    /// `[lo[0..ndim], hi[0..ndim]]`. Queries prune against these, which bounds
    /// the distance to a subtree by the points it actually contains rather
    /// than by the (much looser) space partition of split planes.
    bounds: Vec<f64>,
    n_points: usize,
    ndim: usize,
    leafsize: usize,
}

impl Tree {
    /// Build a tree from a row-major `Vec<f64>` of `ndim`-wide points.
    /// Takes the data by value so the caller can release any Python borrow
    /// before invoking us and we can run under `py.detach`.
    pub fn new(mut data: Vec<f64>, ndim: usize, leafsize: usize) -> Result<Self, KDTreeError> {
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
        // One shared scratch for split keys; recursion hands each child the
        // disjoint half covering its rows, so no per-node allocation happens.
        let mut keys = vec![0.0_f64; n_points];

        // The root bounding-box pass doubles as the finiteness check, so the
        // whole input is only validated once.
        {
            let (lo, hi) = bounds[..2 * ndim].split_at_mut(ndim);
            if !compute_bbox_check_finite(&data, ndim, lo, hi) {
                return Err(KDTreeError::NonFiniteData);
            }
        }

        // Monomorphize the hot recursion over the row width so row swaps,
        // key extraction, and the bbox phase merge compile to straight-line
        // code for the common dimensionalities.
        let build = match ndim {
            1 => build_range::<1>,
            2 => build_range::<2>,
            3 => build_range::<3>,
            4 => build_range::<4>,
            8 => build_range::<8>,
            16 => build_range::<16>,
            _ => build_range::<0>,
        };
        build(
            ndim,
            leafsize,
            &mut data,
            &mut indices,
            &mut nodes,
            &mut bounds,
            &mut keys,
            0,
            ROOT,
            true,
        );

        Ok(Self {
            data,
            indices,
            nodes,
            bounds,
            n_points,
            ndim,
            leafsize,
        })
    }

    /// Reconstruct the original-order data the caller passed to `new`.
    /// Internally we keep the leaf-reordered layout for query cache locality;
    /// the original order is only materialized on demand for the `data` getter.
    pub fn original_data(&self) -> Vec<f64> {
        let mut original = vec![0.0_f64; self.n_points * self.ndim];
        for (pos, &original_idx) in self.indices.iter().enumerate() {
            let src = pos * self.ndim;
            let dst = original_idx as usize * self.ndim;
            original[dst..dst + self.ndim].copy_from_slice(&self.data[src..src + self.ndim]);
        }
        original
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

    pub(crate) fn node(&self, index: u32) -> &Node {
        &self.nodes[index as usize]
    }

    pub(crate) fn points_indexed(&self) -> &[u32] {
        &self.indices
    }

    /// Bounding box of node `index` as `(lo, hi)` slices of length `ndim`.
    pub(crate) fn node_bbox(&self, index: u32) -> (&[f64], &[f64]) {
        let base = 2 * self.ndim * index as usize;
        let pair = &self.bounds[base..base + 2 * self.ndim];
        pair.split_at(self.ndim)
    }

    pub(crate) fn root_bbox(&self) -> (&[f64], &[f64]) {
        self.node_bbox(ROOT)
    }

    /// Return the contiguous slice of coordinates for tree positions
    /// `[start, end)`. Points are physically partitioned during the build, so
    /// this corresponds exactly to the points in the leaf/subtree, row-major.
    pub(crate) fn leaf_block(&self, start: usize, end: usize) -> &[f64] {
        &self.data[start * self.ndim..end * self.ndim]
    }
}

/// Id of the root node: the build lays nodes out in preorder starting here.
pub(crate) const ROOT: u32 = 0;

/// Both children of a flagged node must have shrunk below this fraction of
/// the parent's extent in every non-split dimension for the node to be
/// marked `order_by_box` (see `Node` docs). Genuinely flat data keeps
/// non-split extents near 1.0; data on a low-dimensional manifold halves
/// them at every split.
const BOX_ORDER_MAX_SHRINK_RATIO: f64 = 0.6;

/// Subtrees smaller than this never get the `order_by_box` flag: their
/// extents are noisy order statistics of few samples (which false-positive
/// on flat data), and ordering barely matters that deep anyway.
const BOX_ORDER_MIN_SUBTREE: usize = 64;

/// The split position of a node over `len` points. `count_nodes` and
/// `build_range` both derive the preorder layout from this one policy; they
/// must agree or the layout's slice arithmetic falls apart.
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

/// Build the subtree covering `data` (row-major, `len * ndim`), writing into
/// the preorder `nodes`/`bounds` slices whose first entry is this subtree's
/// root. `base_row` is the global row offset of `data[0]`, `base_node` the
/// global id of `nodes[0]`. Rows are physically partitioned in place, so
/// every recursion level streams contiguous memory and leaves end up
/// contiguous without a separate reorder pass. The preorder layout makes the
/// result identical whether children are built serially or on rayon workers.
///
/// `D` is the compile-time row width (`0` = use `ndim` dynamically).
#[allow(clippy::too_many_arguments)]
fn build_range<const D: usize>(
    ndim: usize,
    leafsize: usize,
    data: &mut [f64],
    indices: &mut [u32],
    nodes: &mut [Node],
    bounds: &mut [f64],
    keys: &mut [f64],
    base_row: usize,
    base_node: u32,
    bbox_ready: bool,
) {
    // The real invariant behind every `D`-specialized helper: the dispatch
    // table in `Tree::new` must map each width to the matching instantiation.
    debug_assert!(D == 0 || D == ndim);
    let ndim = if D > 0 { D } else { ndim };
    let len = indices.len();
    debug_assert_eq!(nodes.len(), count_nodes(len, leafsize));
    let (lo, hi) = bounds[..2 * ndim].split_at_mut(ndim);
    if !bbox_ready {
        compute_bbox::<D>(data, ndim, lo, hi);
    }

    if len <= leafsize {
        nodes[0] = Node::Leaf {
            start: base_row as u32,
            end: (base_row + len) as u32,
        };
        return;
    }

    let split_dim = widest_dimension(lo, hi);
    let mid = split_point(len);

    // Median select runs on a contiguous copy of the split coordinates, then
    // a three-way partition moves whole rows once. This avoids the gather
    // per comparison that dominates an index-indirected select.
    for (key, row) in keys[..len].iter_mut().zip(data.chunks_exact(ndim)) {
        *key = row[split_dim];
    }
    let (_, pivot, _) = keys[..len].select_nth_unstable_by(mid, f64::total_cmp);
    let pivot = *pivot;
    partition_rows::<D>(data, indices, ndim, split_dim, pivot, mid);

    let left_count = count_nodes(mid, leafsize);

    let (left_data, right_data) = data.split_at_mut(mid * ndim);
    let (left_idx, right_idx) = indices.split_at_mut(mid);
    let (node0, child_nodes) = nodes.split_first_mut().expect("subtree has a root node");
    let (left_nodes, right_nodes) = child_nodes.split_at_mut(left_count);
    let (own_bounds, child_bounds) = bounds.split_at_mut(2 * ndim);
    let (left_bounds, right_bounds) = child_bounds.split_at_mut(2 * ndim * left_count);
    let (left_keys, right_keys) = keys.split_at_mut(mid);

    if len >= PARALLEL_BUILD_THRESHOLD {
        rayon::join(
            || {
                build_range::<D>(
                    ndim, leafsize, left_data, left_idx, left_nodes, left_bounds, left_keys,
                    base_row, base_node + 1, false,
                )
            },
            || {
                build_range::<D>(
                    ndim,
                    leafsize,
                    right_data,
                    right_idx,
                    right_nodes,
                    right_bounds,
                    right_keys,
                    base_row + mid,
                    base_node + 1 + left_count as u32,
                    false,
                )
            },
        );
    } else {
        build_range::<D>(
            ndim, leafsize, left_data, left_idx, left_nodes, left_bounds, left_keys, base_row,
            base_node + 1, false,
        );
        build_range::<D>(
            ndim,
            leafsize,
            right_data,
            right_idx,
            right_nodes,
            right_bounds,
            right_keys,
            base_row + mid,
            base_node + 1 + left_count as u32,
            false,
        );
    }

    // With both children's tight boxes known, decide whether queries should
    // order this node's children by box distance during their initial
    // descent (see `Node` docs). Shrinkage along non-split dimensions means
    // the data lives on a lower-dimensional manifold and the split plane
    // alone misjudges proximity.
    let shrunk = |child: &[f64]| {
        let (own_lo, own_hi) = own_bounds.split_at(ndim);
        let (ch_lo, ch_hi) = child[..2 * ndim].split_at(ndim);
        let mut max_ratio = 0.0_f64;
        for d in 0..ndim {
            if d == split_dim {
                continue;
            }
            let parent_extent = own_hi[d] - own_lo[d];
            let ratio = if parent_extent > 0.0 {
                (ch_hi[d] - ch_lo[d]) / parent_extent
            } else {
                1.0
            };
            max_ratio = max_ratio.max(ratio);
        }
        max_ratio < BOX_ORDER_MAX_SHRINK_RATIO
    };
    let order_by_box = ndim > 1
        && len >= BOX_ORDER_MIN_SUBTREE
        && shrunk(left_bounds)
        && shrunk(right_bounds);
    *node0 = Node::inner(
        base_node + 1,
        base_node + 1 + left_count as u32,
        split_dim,
        order_by_box,
        pivot,
    );
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

/// Swap two full rows. With a compile-time width the copy fully unrolls;
/// the dynamic fallback is a single `swap_nonoverlapping` of `ndim`
/// elements.
#[inline(always)]
fn swap_rows<const D: usize>(data: &mut [f64], ndim: usize, a: usize, b: usize) {
    if a == b {
        return;
    }
    let count = if D > 0 { D } else { ndim };
    debug_assert!(a * ndim + count <= data.len() && b * ndim + count <= data.len());
    // SAFETY: `a != b` guarantees the two `count`-element rows are disjoint,
    // and both row ranges are in bounds per the debug assertion above
    // (callers index rows within `data.len() / ndim`).
    unsafe {
        let base = data.as_mut_ptr();
        std::ptr::swap_nonoverlapping(base.add(a * ndim), base.add(b * ndim), count);
    }
}

/// Maximum number of phase accumulators for the flat bbox kernel: covers
/// every row width whose pattern period `ndim / gcd(ndim, LANES)` is at most
/// this many vectors (all of 1..=8, plus common wider even widths).
const MAX_BBOX_PHASES: usize = 8;

/// Tight bounding box of contiguous row-major `data`.
///
/// Instead of chunking each row (useless for rows shorter than a vector),
/// the kernel streams the block as flat `LANES`-wide vectors. Lane `j` of
/// flat vector `i` holds dimension `(i * LANES + j) % ndim`, a pattern that
/// repeats every `ndim / gcd(ndim, LANES)` vectors, so that many phase
/// accumulators cover every dimension; a final scalar merge scatters the
/// accumulator lanes back to their dimensions. Row widths with too long a
/// period fall back to a scalar row sweep.
fn compute_bbox<const D: usize>(data: &[f64], ndim: usize, lo: &mut [f64], hi: &mut [f64]) {
    bbox_kernel::<D, false>(data, ndim, lo, hi);
}

/// `compute_bbox`, plus a fused finiteness check over every element (the
/// `v * 0 != 0` trick catches both infinities and NaNs without extra
/// passes). Returns `false` when any element is non-finite.
fn compute_bbox_check_finite(data: &[f64], ndim: usize, lo: &mut [f64], hi: &mut [f64]) -> bool {
    bbox_kernel::<0, true>(data, ndim, lo, hi)
}

/// `D` is the compile-time row width (`0` = dynamic): with it fixed, the
/// phase count is a constant and the whole phase/merge machinery unrolls.
fn bbox_kernel<const D: usize, const CHECK_FINITE: bool>(
    data: &[f64],
    ndim: usize,
    lo: &mut [f64],
    hi: &mut [f64],
) -> bool {
    debug_assert!(D == 0 || D == ndim);
    let ndim = if D > 0 { D } else { ndim };
    let lo = &mut lo[..ndim];
    let hi = &mut hi[..ndim];
    lo.fill(f64::INFINITY);
    hi.fill(f64::NEG_INFINITY);

    let phases = ndim / gcd(ndim, LANES);
    let mut nonfinite = Mask::<i64, LANES>::splat(false);
    if phases <= MAX_BBOX_PHASES {
        let mut acc_lo = [F64s::splat(f64::INFINITY); MAX_BBOX_PHASES];
        let mut acc_hi = [F64s::splat(f64::NEG_INFINITY); MAX_BBOX_PHASES];
        let (chunks, rest) = data.as_chunks::<LANES>();
        let mut phase = 0;
        for c in chunks {
            let v = F64s::from_array(*c);
            acc_lo[phase] = vmin(v, acc_lo[phase]);
            acc_hi[phase] = vmax(v, acc_hi[phase]);
            if CHECK_FINITE {
                nonfinite |= nonfinite_lanes(v);
            }
            phase += 1;
            if phase == phases {
                phase = 0;
            }
        }
        for (i, (al, ah)) in acc_lo[..phases].iter().zip(&acc_hi[..phases]).enumerate() {
            for j in 0..LANES {
                let dim = (i * LANES + j) % ndim;
                lo[dim] = lo[dim].min(al[j]);
                hi[dim] = hi[dim].max(ah[j]);
            }
        }
        let mut dim = (chunks.len() * LANES) % ndim;
        for &v in rest {
            lo[dim] = lo[dim].min(v);
            hi[dim] = hi[dim].max(v);
            if CHECK_FINITE && !v.is_finite() {
                return false;
            }
            dim += 1;
            if dim == ndim {
                dim = 0;
            }
        }
    } else {
        for coords in data.chunks_exact(ndim) {
            for ((l, h), &v) in lo.iter_mut().zip(hi.iter_mut()).zip(coords) {
                *l = l.min(v);
                *h = h.max(v);
                if CHECK_FINITE && !v.is_finite() {
                    return false;
                }
            }
        }
    }
    !CHECK_FINITE || !nonfinite.any()
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
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
        let result = Tree::new(Vec::new(), 2, 32);
        assert!(result.is_err());
    }

    #[test]
    fn build_preserves_shape_information() {
        let data = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
        let tree = Tree::new(data, 2, 2).expect("tree should build");

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
        let tree = Tree::new(data.clone(), 2, 1).expect("tree should build");
        assert_eq!(tree.original_data(), data);
    }

    #[test]
    fn build_allocation_count_does_not_scale_with_n() {
        let n = 1000;
        let ndim = 8;
        let leafsize = 16;
        let data: Vec<f64> = (0..n * ndim).map(|i| (i as f64) * 0.001).collect();

        let info = allocation_counter::measure(move || {
            Tree::new(data, ndim, leafsize).expect("tree should build");
        });

        assert!(
            info.count_total < 20,
            "expected < 20 allocations, got {}",
            info.count_total
        );
    }
}
