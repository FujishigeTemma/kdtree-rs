//! k-nearest-neighbor queries: validation, the branch-and-bound descent,
//! and the running k-best set. Every distance-valued `f64` in this module
//! is a *reduced distance* (see `metric.rs`); only `Best::write_results`
//! restores true distances, once per emitted result.
//!
//! # Pruning
//!
//! The descent maintains an O(1)-updatable lower bound on the reduced
//! distance from the query to the current cell (per-axis parts in
//! [`Scratch::cell`], the folded total passed down the recursion). Before
//! entering a far child two bounds are checked in order of cost: that
//! incremental split-plane bound, then — only if it fails to prune — the
//! tight bounding box of the points the far subtree actually contains. The
//! tight box is what collapses degenerate clustered data (where every split
//! lands on the same few dimensions and plane bounds stay uselessly small);
//! the plane bound keeps the common well-separated descent free of O(ndim)
//! box work.

use rayon::prelude::*;

use crate::error::KDTreeError;
use crate::kernel;
use crate::metric::{Metric, box_axis_offset};
use crate::tree::{Node, ROOT, Tree};

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
        if k == 0 {
            return Err(KDTreeError::InvalidK);
        }
        if queries.is_empty() || !queries.len().is_multiple_of(self.ndim()) {
            return Err(KDTreeError::InvalidShape(
                "queries must be a contiguous row-major matrix",
            ));
        }
        if !kernel::all_finite(queries) {
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
            cutoff: metric.reduce(max_distance),
            eps_factor: metric.eps_factor(eps),
            metric,
        };

        let n_queries = queries.len() / self.ndim();
        let mut distances = vec![0.0_f64; n_queries * k];
        let mut indices = vec![0_i64; n_queries * k];

        // Monomorphizing over the metric as well was measured slower here
        // (the const-folded kernels inline into `descend` and bloat it);
        // the enum dispatch stays perfectly predicted instead.
        if k == 1 {
            self.run_queries::<Best1>(queries, k, params, parallel, &mut distances, &mut indices);
        } else {
            self.run_queries::<BestK>(queries, k, params, parallel, &mut distances, &mut indices);
        }

        Ok((distances, indices))
    }

    fn run_queries<B: Best>(
        &self,
        queries: &[f64],
        k: usize,
        params: QueryParams,
        parallel: bool,
        distances: &mut [f64],
        indices: &mut [i64],
    ) {
        let ndim = self.ndim();
        let n_queries = queries.len() / ndim;
        let run = |scratch: &mut Scratch<B>, i: usize, out_d: &mut [f64], out_i: &mut [i64]| {
            let descent = Descent {
                tree: self,
                params,
                q: &queries[i * ndim..(i + 1) * ndim],
            };
            descent.run(scratch, out_d, out_i);
        };

        if parallel && n_queries > 1 {
            distances
                .par_chunks_mut(k)
                .zip(indices.par_chunks_mut(k))
                .enumerate()
                .with_min_len(PARALLEL_QUERY_MIN_CHUNK)
                .for_each_init(
                    || Scratch::<B>::new(ndim, k),
                    |scratch, (i, (out_d, out_i))| run(scratch, i, out_d, out_i),
                );
        } else {
            let mut scratch = Scratch::<B>::new(ndim, k);
            distances
                .chunks_mut(k)
                .zip(indices.chunks_mut(k))
                .enumerate()
                .for_each(|(i, (out_d, out_i))| run(&mut scratch, i, out_d, out_i));
        }
    }
}

/// Immutable parameters shared by every query of one `query` call.
/// `cutoff` is `max_distance` in the reduced domain.
#[derive(Clone, Copy)]
struct QueryParams {
    cutoff: f64,
    eps_factor: f64,
    metric: Metric,
}

/// One thread's reusable mutable state, allocated once and reused across
/// queries. Kept separate from [`Descent`] so the recursion borrows its
/// read-only context and its mutable state independently — handing the hot
/// path one `&mut` over everything forces the compiler to reload the
/// context after every recursive call.
struct Scratch<B> {
    /// Per-axis reduced contribution of the current cell's lower bound.
    ///
    /// Invariant between queries: all zeros — the correct seed for a query
    /// inside the root box, which is the overwhelmingly common case. A rare
    /// out-of-box query fills it via `Descent::seed_cell` and restores the
    /// zeros after its descent.
    cell: Vec<f64>,
    best: B,
}

impl<B: Best> Scratch<B> {
    fn new(ndim: usize, k: usize) -> Self {
        Self {
            cell: vec![0.0; ndim],
            best: B::new(k),
        }
    }
}

/// The immutable context of one query's descent: the tree, the search
/// parameters, and the query point.
#[derive(Clone, Copy)]
struct Descent<'a> {
    tree: &'a Tree,
    params: QueryParams,
    q: &'a [f64],
}

impl<'a> Descent<'a> {
    /// Answer this query, writing `k` results to `out_d`/`out_i`.
    fn run<B: Best>(&self, s: &mut Scratch<B>, out_d: &mut [f64], out_i: &mut [i64]) {
        let QueryParams {
            cutoff,
            eps_factor,
            metric,
        } = self.params;
        s.best.reset();
        let (lo, hi) = self.tree.root_box();
        if kernel::box_rd(metric, self.q, lo, hi) == 0.0 {
            self.descend(ROOT, 0.0, s);
        } else {
            let seed = self.seed_cell(lo, hi, &mut s.cell);
            if seed * eps_factor <= cutoff {
                self.descend(ROOT, seed, s);
            }
            s.cell.fill(0.0);
        }
        s.best
            .write_results(out_d, out_i, self.tree.n_points(), metric);
    }

    /// Fill `cell` for a query outside the root box and return the folded
    /// cell bound.
    fn seed_cell(&self, lo: &[f64], hi: &[f64], cell: &mut [f64]) -> f64 {
        let metric = self.params.metric;
        let mut rd = 0.0_f64;
        for d in 0..self.q.len() {
            let axis = metric.axis_rd(box_axis_offset(self.q[d], lo[d], hi[d]));
            cell[d] = axis;
            rd = metric.fold(rd, axis);
        }
        rd
    }

    /// Is a subtree whose lower bound is `rd` still worth entering? The
    /// approximation factor scales the lower bound rather than the k-best
    /// bound so the comparison stays in the reduced domain.
    #[inline(always)]
    fn admits<B: Best>(&self, rd: f64, s: &Scratch<B>) -> bool {
        rd * self.params.eps_factor <= s.best.bound(self.params.cutoff)
    }

    /// Recursive branch-and-bound descent into `node_id`, whose cell has the
    /// lower bound `cell_rd` (per-axis parts in `s.cell`).
    fn descend<B: Best>(&self, node_id: u32, cell_rd: f64, s: &mut Scratch<B>) {
        let QueryParams {
            cutoff,
            eps_factor,
            metric,
        } = self.params;

        match *self.tree.node(node_id) {
            Node::Leaf { start, end } => {
                self.scan_leaf(start as usize, end as usize, s);
            }
            Node::Inner {
                left,
                right,
                split_dim,
                order_by_box,
                split_value,
            } => {
                let dim = split_dim as usize;
                let diff = self.q[dim] - split_value;
                let (near, far) = if diff <= 0.0 {
                    (left, right)
                } else {
                    (right, left)
                };

                let far_axis = metric.axis_rd(diff.abs());
                let far_rd = metric.replace_axis(cell_rd, s.cell[dim], far_axis);

                if order_by_box {
                    self.descend_by_box(near, far, dim, cell_rd, far_axis, far_rd, s);
                    return;
                }

                self.descend(near, cell_rd, s);

                let bound = s.best.bound(cutoff);
                if far_rd * eps_factor <= bound {
                    let (far_lo, far_hi) = self.tree.box_of(far);
                    if kernel::box_rd(metric, self.q, far_lo, far_hi) * eps_factor <= bound {
                        self.enter_far(far, dim, far_axis, far_rd, s);
                    }
                }
            }
        }
    }

    /// Visit both children of a manifold-clustered node (see [`Node`] docs):
    /// the split plane misjudges both proximity and pruning there, so the
    /// visit is ordered and both children gated by tight box distance.
    /// Kept out of line so the flag test stays cheap in `descend`'s hot
    /// path — flat data never sets the flag and pays nothing beyond it.
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn descend_by_box<B: Best>(
        &self,
        near: u32,
        far: u32,
        dim: usize,
        cell_rd: f64,
        far_axis: f64,
        far_rd: f64,
        s: &mut Scratch<B>,
    ) {
        let metric = self.params.metric;
        let (n_lo, n_hi) = self.tree.box_of(near);
        let near_box = kernel::box_rd(metric, self.q, n_lo, n_hi);
        let (f_lo, f_hi) = self.tree.box_of(far);
        let far_box = kernel::box_rd(metric, self.q, f_lo, f_hi);

        // The near child keeps the parent's incremental cell bound: box
        // distances gate the visits but never leak into the plane-bound
        // algebra, which stays consistent with `cell`. Both gates are
        // re-tested as we go, since visiting one child tightens the bound
        // for the other.
        if near_box <= far_box {
            if self.admits(near_box, s) {
                self.descend(near, cell_rd, s);
            }
            if self.admits(far_box, s) {
                self.enter_far(far, dim, far_axis, far_rd, s);
            }
        } else {
            if self.admits(far_box, s) {
                self.enter_far(far, dim, far_axis, far_rd, s);
            }
            if self.admits(near_box, s) {
                self.descend(near, cell_rd, s);
            }
        }
    }

    /// Enter the plane-far child: apply the far side's axis contribution to
    /// `cell`, descend with the updated bound, and restore. The one place
    /// that owns the save/restore protocol of `cell`.
    #[inline(always)]
    fn enter_far<B: Best>(
        &self,
        far: u32,
        dim: usize,
        far_axis: f64,
        far_rd: f64,
        s: &mut Scratch<B>,
    ) {
        let saved_axis = s.cell[dim];
        s.cell[dim] = far_axis;
        self.descend(far, far_rd, s);
        s.cell[dim] = saved_axis;
    }

    /// Evaluate every point of a leaf against the current k-best set. The
    /// scan kernel owns the strategy (SIMD block kernels for short rows,
    /// early-exit per-point scan otherwise); this only supplies the bound
    /// and folds hits into the k-best buffer.
    #[inline]
    fn scan_leaf<B: Best>(&self, start: usize, end: usize, s: &mut Scratch<B>) {
        let QueryParams { cutoff, metric, .. } = self.params;
        let block = self.tree.rows(start, end);
        let originals = &self.tree.indices;
        let best = &mut s.best;
        let bound = best.bound(cutoff);
        kernel::scan_leaf(metric, self.q, block, bound, |offset, rd| {
            best.consider(rd, originals[start + offset]);
            best.bound(cutoff)
        });
    }
}

/// Running k-best set of one query. Monomorphizing the descent over this
/// keeps the ubiquitous `k == 1` case in two registers instead of heap
/// buffers.
trait Best {
    fn new(k: usize) -> Self;
    fn reset(&mut self);
    /// Current pruning bound: the k-th best reduced distance so far, capped
    /// at `cutoff`.
    fn bound(&self, cutoff: f64) -> f64;
    /// Offer `(rd, idx)`. Ties resolve toward the smaller original index,
    /// matching `numpy.argsort(kind="stable")`.
    fn consider(&mut self, rd: f64, idx: u32);
    fn write_results(&self, out_d: &mut [f64], out_i: &mut [i64], n_points: usize, m: Metric);
}

/// Single nearest neighbor.
struct Best1 {
    rd: f64,
    i: u32,
}

impl Best for Best1 {
    fn new(_k: usize) -> Self {
        Self {
            rd: f64::INFINITY,
            i: u32::MAX,
        }
    }

    fn reset(&mut self) {
        self.rd = f64::INFINITY;
        self.i = u32::MAX;
    }

    #[inline]
    fn bound(&self, cutoff: f64) -> f64 {
        self.rd.min(cutoff)
    }

    #[inline]
    fn consider(&mut self, rd: f64, idx: u32) {
        if rd < self.rd || (rd == self.rd && idx < self.i) {
            self.rd = rd;
            self.i = idx;
        }
    }

    fn write_results(&self, out_d: &mut [f64], out_i: &mut [i64], n_points: usize, m: Metric) {
        if self.i != u32::MAX {
            out_d[0] = m.restore(self.rd);
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
    rds: Vec<f64>,
    ids: Vec<u32>,
}

impl Best for BestK {
    fn new(k: usize) -> Self {
        Self {
            k,
            rds: Vec::with_capacity(k),
            ids: Vec::with_capacity(k),
        }
    }

    fn reset(&mut self) {
        self.rds.clear();
        self.ids.clear();
    }

    #[inline]
    fn bound(&self, cutoff: f64) -> f64 {
        if self.rds.len() < self.k {
            cutoff
        } else {
            self.rds[self.k - 1].min(cutoff)
        }
    }

    #[inline]
    fn consider(&mut self, rd: f64, idx: u32) {
        if self.rds.len() == self.k {
            let worst_rd = self.rds[self.k - 1];
            if rd > worst_rd || (rd == worst_rd && self.ids[self.k - 1] <= idx) {
                return;
            }
            self.rds.pop();
            self.ids.pop();
        }
        let mut pos = self.rds.len();
        while pos > 0 {
            let prev_rd = self.rds[pos - 1];
            if prev_rd < rd || (prev_rd == rd && self.ids[pos - 1] < idx) {
                break;
            }
            pos -= 1;
        }
        self.rds.insert(pos, rd);
        self.ids.insert(pos, idx);
    }

    fn write_results(&self, out_d: &mut [f64], out_i: &mut [i64], n_points: usize, m: Metric) {
        for j in 0..out_d.len() {
            if j < self.rds.len() {
                out_d[j] = m.restore(self.rds[j]);
                out_i[j] = self.ids[j] as i64;
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
        let tree = Tree::new(data, 2, 2, true).expect("tree should build");

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
        let tree = Tree::new(data, 2, 1, true).expect("tree should build");

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
        let tree = Tree::new(data, 2, 1, true).expect("tree should build");
        let (d, i) = tree
            .query(&[0.9, 0.9], 1, 2.0, None, 0.0, false)
            .expect("query should succeed");
        assert_eq!(i, vec![1]);
        assert_relative_eq!(d[0], (0.02_f64).sqrt());
    }
}
