use rayon::prelude::*;

use crate::error::KDTreeError;
use crate::metric::{Metric, box_axis_offset};
use crate::node::{Node, unpack_split};
use crate::tree::{ROOT, Tree};

/// Minimum queries per rayon task. Each query can be well under a
/// microsecond, so without a floor the per-task overhead of a
/// one-query-per-item split swamps the work being distributed.
const PARALLEL_QUERY_MIN_CHUNK: usize = 16;

impl Tree {
    pub fn query(
        &self,
        queries: &[f64],
        k: usize,
        p: f64,
        max_distance: Option<f64>,
        eps: f64,
        parallel: bool,
    ) -> Result<(Vec<f64>, Vec<i64>), KDTreeError> {
        let ndim = self.ndim();
        if k == 0 {
            return Err(KDTreeError::InvalidK);
        }
        if queries.is_empty() || !queries.len().is_multiple_of(ndim) {
            return Err(KDTreeError::InvalidShape(
                "queries must be a contiguous row-major matrix",
            ));
        }
        if !crate::simd::all_finite(queries) {
            return Err(KDTreeError::NonFiniteData);
        }
        if !eps.is_finite() || eps < 0.0 {
            return Err(KDTreeError::InvalidEps(eps));
        }
        let max_distance = max_distance.unwrap_or(f64::INFINITY);
        if !max_distance.is_infinite() && (!max_distance.is_finite() || max_distance < 0.0) {
            return Err(KDTreeError::InvalidMaxDistance(max_distance));
        }

        let metric = Metric::new(p)?;
        let params = QueryParams {
            limit: metric.to_accum(max_distance),
            eps_factor: metric.eps_factor(eps),
            metric,
        };
        let n_queries = queries.len() / ndim;

        let mut distances = vec![0.0_f64; n_queries * k];
        let mut indices = vec![0_i64; n_queries * k];

        // Monomorphizing over the metric as well was measured slower here
        // (the const-folded kernels inline into `descend` and bloat it);
        // the enum dispatch stays perfectly predicted instead.
        if k == 1 {
            self.run_queries::<Best1>(queries, k, &params, parallel, &mut distances, &mut indices);
        } else {
            self.run_queries::<BestK>(queries, k, &params, parallel, &mut distances, &mut indices);
        }

        Ok((distances, indices))
    }

    fn run_queries<B: Best>(
        &self,
        queries: &[f64],
        k: usize,
        params: &QueryParams,
        parallel: bool,
        distances: &mut [f64],
        indices: &mut [i64],
    ) {
        let ndim = self.ndim();
        let n_points = self.n_points();
        let metric = params.metric;

        let run = |state: &mut (DescentState, B), q_idx: usize, out_d: &mut [f64], out_i: &mut [i64]| {
            let (cell, best) = state;
            best.reset();
            let q = &queries[q_idx * ndim..(q_idx + 1) * ndim];
            let (lo, hi) = self.root_bbox();
            // Fast path: queries inside the root box (the overwhelmingly
            // common case) need no per-axis seed at all — `cell` upholds
            // "all zeros between queries", which is exactly the seed.
            if metric.bbox_accum(q, lo, hi) == 0.0 {
                cell.min_dist = 0.0;
                self.descend(ROOT, q, params, cell, best);
            } else {
                cell.seed_from_root(q, (lo, hi), metric);
                if cell.min_dist * params.eps_factor <= params.limit {
                    self.descend(ROOT, q, params, cell, best);
                }
                cell.clear();
            }
            best.write_results(out_d, out_i, n_points, metric);
        };

        let n_queries = queries.len() / ndim;
        if parallel && n_queries > 1 {
            distances
                .par_chunks_mut(k)
                .zip(indices.par_chunks_mut(k))
                .enumerate()
                .with_min_len(PARALLEL_QUERY_MIN_CHUNK)
                .for_each_init(
                    || (DescentState::new(ndim), B::new(k)),
                    |state, (q_idx, (d_chunk, i_chunk))| run(state, q_idx, d_chunk, i_chunk),
                );
        } else {
            let mut state = (DescentState::new(ndim), B::new(k));
            distances
                .chunks_mut(k)
                .zip(indices.chunks_mut(k))
                .enumerate()
                .for_each(|(q_idx, (d_chunk, i_chunk))| run(&mut state, q_idx, d_chunk, i_chunk));
        }
    }

    /// Recursive branch-and-bound descent. The near child is entered with an
    /// O(1) incremental cell bound (`cell.min_dist`, per-axis parts in
    /// `cell.side`). Before entering the far child two bounds are checked in
    /// order of cost: the incremental split-plane bound, then — only if that
    /// fails to prune — the tight bounding box of the points the far subtree
    /// actually contains. The tight box is what collapses degenerate
    /// clustered data (where every split lands on the same few dimensions
    /// and plane bounds stay uselessly small); the plane bound keeps the
    /// common well-separated descent free of O(ndim) box work.
    fn descend<B: Best>(
        &self,
        node_id: u32,
        q: &[f64],
        params: &QueryParams,
        cell: &mut DescentState,
        best: &mut B,
    ) {
        let QueryParams {
            limit,
            eps_factor,
            metric,
        } = *params;

        match *self.node(node_id) {
            Node::Leaf { start, end } => {
                self.scan_leaf(start as usize, end as usize, q, params, best);
            }
            Node::Inner {
                left,
                right,
                split_dim,
                split_value,
            } => {
                let (dim, order_by_box) = unpack_split(split_dim);
                let diff = q[dim] - split_value;
                let (near, far) = if diff <= 0.0 { (left, right) } else { (right, left) };

                let new_axis = metric.axis_accum(diff.abs());
                let old_axis = cell.side[dim];
                let new_min = metric.replace_axis(cell.min_dist, old_axis, new_axis);

                if order_by_box {
                    self.descend_box_ordered(
                        near, far, dim, new_axis, old_axis, new_min, q, params, cell, best,
                    );
                    return;
                }

                self.descend(near, q, params, cell, best);

                let upper = best.upper(limit);
                if new_min * eps_factor <= upper {
                    let (far_lo, far_hi) = self.node_bbox(far);
                    if metric.bbox_accum(q, far_lo, far_hi) * eps_factor <= upper {
                        self.enter_far(far, dim, new_axis, old_axis, new_min, q, params, cell, best);
                    }
                }
            }
        }
    }

    /// Visit both children of a manifold-clustered node (see `Node` docs):
    /// the split plane misjudges both proximity and pruning there, so the
    /// visit is ordered and both children gated by tight box distance.
    /// Kept out of line so the flag test stays cheap in `descend`'s hot
    /// path — flat data never sets the flag and pays nothing beyond it.
    #[allow(clippy::too_many_arguments)]
    #[inline(never)]
    fn descend_box_ordered<B: Best>(
        &self,
        near: u32,
        far: u32,
        dim: usize,
        new_axis: f64,
        old_axis: f64,
        new_min: f64,
        q: &[f64],
        params: &QueryParams,
        cell: &mut DescentState,
        best: &mut B,
    ) {
        let QueryParams {
            limit,
            eps_factor,
            metric,
        } = *params;
        let (n_lo, n_hi) = self.node_bbox(near);
        let d_near = metric.bbox_accum(q, n_lo, n_hi);
        let (f_lo, f_hi) = self.node_bbox(far);
        let d_far = metric.bbox_accum(q, f_lo, f_hi);

        if d_near <= d_far {
            if d_near * eps_factor <= best.upper(limit) {
                self.descend(near, q, params, cell, best);
            }
            if d_far * eps_factor <= best.upper(limit) {
                self.enter_far(far, dim, new_axis, old_axis, new_min, q, params, cell, best);
            }
        } else {
            if d_far * eps_factor <= best.upper(limit) {
                self.enter_far(far, dim, new_axis, old_axis, new_min, q, params, cell, best);
            }
            if d_near * eps_factor <= best.upper(limit) {
                self.descend(near, q, params, cell, best);
            }
        }
    }

    /// Enter the plane-far child: apply the far side's axis contribution to
    /// the incremental cell state, descend, and restore. The one place that
    /// owns the save/restore protocol of `DescentState`.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn enter_far<B: Best>(
        &self,
        far: u32,
        dim: usize,
        new_axis: f64,
        old_axis: f64,
        new_min: f64,
        q: &[f64],
        params: &QueryParams,
        cell: &mut DescentState,
        best: &mut B,
    ) {
        let saved_min = cell.min_dist;
        cell.side[dim] = new_axis;
        cell.min_dist = new_min;
        self.descend(far, q, params, cell, best);
        cell.side[dim] = old_axis;
        cell.min_dist = saved_min;
    }

    /// Evaluate every point of a leaf against the current k-best set. The
    /// metric owns the scan strategy (SIMD block kernels for short rows,
    /// early-exit per-point scan otherwise); this only supplies the bound
    /// and folds hits into the k-best buffer.
    #[inline]
    fn scan_leaf<B: Best>(
        &self,
        start: usize,
        end: usize,
        q: &[f64],
        params: &QueryParams,
        best: &mut B,
    ) {
        let &QueryParams { limit, metric, .. } = params;
        let block = self.leaf_block(start, end);
        let originals = self.points_indexed();
        let bound = best.upper(limit);
        metric.scan_block(q, block, bound, |offset, d| {
            best.consider(d, originals[start + offset]);
            best.upper(limit)
        });
    }
}

/// Immutable parameters shared by every step of one `query` call.
#[derive(Clone, Copy)]
struct QueryParams {
    limit: f64,
    eps_factor: f64,
    metric: Metric,
}

/// Incremental cell state of the descent: the per-axis distance contribution
/// of the current cell and its accumulated L^p lower bound.
///
/// Invariant between queries: `side` is all zeros — the seed for a query
/// inside the root box. A rare out-of-box query fills it via
/// `seed_from_root` and the caller restores the zeros with `clear` after
/// the descent.
struct DescentState {
    side: Vec<f64>,
    min_dist: f64,
}

impl DescentState {
    fn new(ndim: usize) -> Self {
        Self {
            side: vec![0.0; ndim],
            min_dist: 0.0,
        }
    }

    fn clear(&mut self) {
        self.side.fill(0.0);
        self.min_dist = 0.0;
    }

    fn seed_from_root(&mut self, q: &[f64], bbox: (&[f64], &[f64]), metric: Metric) {
        let (lo, hi) = bbox;
        let mut acc = 0.0_f64;
        for d in 0..q.len() {
            let axis = metric.axis_accum(box_axis_offset(q[d], lo[d], hi[d]));
            self.side[d] = axis;
            acc = metric.fold_axis(acc, axis);
        }
        self.min_dist = acc;
    }
}

/// Running k-best set of one query. Monomorphizing `descend` over this keeps
/// the ubiquitous `k == 1` case in two registers instead of heap buffers.
trait Best {
    fn new(k: usize) -> Self;
    fn reset(&mut self);
    /// Current pruning bound: the k-th best distance so far, capped at `limit`.
    fn upper(&self, limit: f64) -> f64;
    /// Offer `(d, idx)`. Ties resolve toward the smaller original index,
    /// matching `numpy.argsort(kind="stable")`.
    fn consider(&mut self, d: f64, idx: u32);
    fn write_results(&self, out_d: &mut [f64], out_i: &mut [i64], n_points: usize, m: Metric);
}

/// Single nearest neighbor.
struct Best1 {
    d: f64,
    i: u32,
}

impl Best for Best1 {
    fn new(_k: usize) -> Self {
        Self {
            d: f64::INFINITY,
            i: u32::MAX,
        }
    }

    fn reset(&mut self) {
        self.d = f64::INFINITY;
        self.i = u32::MAX;
    }

    #[inline]
    fn upper(&self, limit: f64) -> f64 {
        self.d.min(limit)
    }

    #[inline]
    fn consider(&mut self, d: f64, idx: u32) {
        if d < self.d || (d == self.d && idx < self.i) {
            self.d = d;
            self.i = idx;
        }
    }

    fn write_results(&self, out_d: &mut [f64], out_i: &mut [i64], n_points: usize, m: Metric) {
        if self.i != u32::MAX {
            out_d[0] = m.finish(self.d);
            out_i[0] = self.i as i64;
        } else {
            out_d[0] = f64::INFINITY;
            out_i[0] = n_points as i64;
        }
    }
}

/// General `k`: a small sorted buffer, insertion-sorted on the fly.
struct BestK {
    k: usize,
    nb_d: Vec<f64>,
    nb_i: Vec<u32>,
}

impl Best for BestK {
    fn new(k: usize) -> Self {
        Self {
            k,
            nb_d: Vec::with_capacity(k),
            nb_i: Vec::with_capacity(k),
        }
    }

    fn reset(&mut self) {
        self.nb_d.clear();
        self.nb_i.clear();
    }

    #[inline]
    fn upper(&self, limit: f64) -> f64 {
        if self.nb_d.len() < self.k {
            limit
        } else {
            self.nb_d[self.k - 1].min(limit)
        }
    }

    #[inline]
    fn consider(&mut self, d: f64, idx: u32) {
        if self.nb_d.len() == self.k {
            let worst_d = self.nb_d[self.k - 1];
            if d > worst_d || (d == worst_d && self.nb_i[self.k - 1] <= idx) {
                return;
            }
            self.nb_d.pop();
            self.nb_i.pop();
        }
        let mut pos = self.nb_d.len();
        while pos > 0 {
            let prev_d = self.nb_d[pos - 1];
            if prev_d < d || (prev_d == d && self.nb_i[pos - 1] < idx) {
                break;
            }
            pos -= 1;
        }
        self.nb_d.insert(pos, d);
        self.nb_i.insert(pos, idx);
    }

    fn write_results(&self, out_d: &mut [f64], out_i: &mut [i64], n_points: usize, m: Metric) {
        for j in 0..out_d.len() {
            if j < self.nb_d.len() {
                out_d[j] = m.finish(self.nb_d[j]);
                out_i[j] = self.nb_i[j] as i64;
            } else {
                out_d[j] = f64::INFINITY;
                out_i[j] = n_points as i64;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use crate::tree::Tree;

    #[test]
    fn query_returns_exact_nearest_neighbors() {
        let data = vec![0.0, 0.0, 2.0, 0.0, 4.0, 0.0, 5.0, 0.0];
        let tree = Tree::new(data, 2, 2).expect("tree should build");

        let (distances, indices) = tree
            .query(&[1.5, 0.0], 2, 2.0, None, 0.0, false)
            .expect("query should succeed");

        assert_eq!(indices, vec![1, 0]);
        assert_relative_eq!(distances[0], 0.5);
        assert_relative_eq!(distances[1], 1.5);
    }

    #[test]
    fn query_pads_missing_neighbors() {
        let data = vec![0.0, 0.0, 10.0, 0.0];
        let tree = Tree::new(data, 2, 1).expect("tree should build");

        let (distances, indices) = tree
            .query(&[0.0, 0.0, 11.0, 0.0], 3, 2.0, Some(2.0), 0.0, false)
            .expect("query should succeed");

        assert_eq!(indices.len(), 6);
        assert_eq!(indices[2], 2);
        assert!(distances[2].is_infinite());
        assert_eq!(indices[5], 2);
        assert!(distances[5].is_infinite());
    }

    #[test]
    fn single_best_query_matches_k1() {
        let data = vec![0.0, 0.0, 1.0, 1.0, -2.0, 0.5, 3.0, -1.0];
        let tree = Tree::new(data, 2, 1).expect("tree should build");
        let (d, i) = tree
            .query(&[0.9, 0.9], 1, 2.0, None, 0.0, false)
            .expect("query should succeed");
        assert_eq!(i, vec![1]);
        assert_relative_eq!(d[0], (0.02_f64).sqrt());
    }
}
