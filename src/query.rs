//! The descent prunes with two bounds, checked in order of cost: an
//! O(1)-updatable lower bound from the query to the current cell (per-axis parts
//! in [`CellBound`], the folded total passed down the recursion), and — only when
//! that fails to prune — the tight box of the points the subtree contains. The
//! tight box is what collapses degenerate clustered data, where plane bounds stay
//! uselessly small; the plane bound keeps the common well-separated descent free
//! of O(ndim) box work.
//!
//! **Box distances gate visits and nothing else.** A child is always entered with
//! a bound derived from [`CellBound`], never with its own box distance. Feed a box
//! distance into the recursion and every later `replace_axis` subtracts a
//! contribution that is not in the total — silently wrong results, not a crash.

use std::marker::PhantomData;

use rayon::prelude::*;

use crate::error::KDTreeError;
use crate::kernel::{self, Strategy, with_plan};
use crate::layout::{BBox, axis_offset};
use crate::metric::{Dist, Metric};
use crate::tree::{Node, ROOT, Tree};

/// A query can be well under a microsecond, so without a floor rayon's per-task
/// overhead swamps the work being distributed.
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
        if max_distance.is_nan() || max_distance < 0.0 {
            return Err(KDTreeError::InvalidMaxDistance(max_distance));
        }

        let metric = Metric::new(p)?;
        let params = QueryParams {
            cutoff: metric.reduce(max_distance),
            eps_factor: metric.reduce(1.0 + eps).get(),
            metric,
        };

        let n_queries = queries.len() / self.ndim();
        let mut distances = vec![0.0_f64; n_queries * k];
        let mut indices = vec![0_i64; n_queries * k];

        // Monomorphizing over the metric as well was measured slower: the
        // const-folded kernels inline into `descend` and bloat it, where the
        // enum dispatch stays perfectly predicted.
        with_plan!(metric, self.ndim(), |S| if k == 1 {
            self.run::<Best1, S>(queries, k, params, parallel, &mut distances, &mut indices)
        } else {
            self.run::<BestK, S>(queries, k, params, parallel, &mut distances, &mut indices)
        });

        Ok((distances, indices))
    }

    #[allow(clippy::too_many_arguments)]
    fn run<B: Best, K: Strategy>(
        &self,
        queries: &[f64],
        k: usize,
        params: QueryParams,
        parallel: bool,
        distances: &mut [f64],
        indices: &mut [i64],
    ) {
        let ndim = self.ndim();
        let n_points = self.n_points();
        let n_queries = queries.len() / ndim;
        let run = |scratch: &mut Scratch<B>, i: usize, out_d: &mut [f64], out_i: &mut [i64]| {
            let descent = Descent::<K> {
                tree: self,
                n_points,
                params,
                q: &queries[i * ndim..(i + 1) * ndim],
                strategy: PhantomData,
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

#[derive(Clone, Copy)]
struct QueryParams {
    cutoff: Dist,
    eps_factor: f64,
    metric: Metric,
}

/// Kept separate from [`Descent`] so the recursion borrows read-only context and
/// mutable state independently — one `&mut` over both forces a context reload
/// after every recursive call.
struct Scratch<B> {
    cell: CellBound,
    best: B,
}

impl<B: Best> Scratch<B> {
    fn new(ndim: usize, k: usize) -> Self {
        Self {
            cell: CellBound::new(ndim),
            best: B::new(k),
        }
    }
}

/// The split-plane lower bound, kept as its per-axis contributions — the
/// breakdown that makes `replace_axis` an O(1) update. The folded total travels
/// as a recursion argument instead, being per-frame.
///
/// Invariant: all axes are zero between queries. That is already the answer for a
/// query inside the root box, the common case, so [`CellBound::start`] normally
/// writes nothing and [`CellBound::finish`] has nothing to undo.
struct CellBound {
    axes: Vec<Dist>,
    seeded: bool,
}

impl CellBound {
    fn new(ndim: usize) -> Self {
        Self {
            axes: vec![Dist::ZERO; ndim],
            seeded: false,
        }
    }

    #[inline(always)]
    fn start(&mut self, m: Metric, q: &[f64], root: BBox<'_>) -> Dist {
        if kernel::box_dist(m, q, root) == Dist::ZERO {
            return Dist::ZERO;
        }
        self.seed(m, q, root)
    }

    fn seed(&mut self, m: Metric, q: &[f64], root: BBox<'_>) -> Dist {
        let mut dist = Dist::ZERO;
        let bounds = q.iter().zip(root.lo).zip(root.hi);
        for (slot, ((&q, &lo), &hi)) in self.axes.iter_mut().zip(bounds) {
            let axis = m.reduce(axis_offset(q, lo, hi));
            *slot = axis;
            dist = m.fold(dist, axis);
        }
        self.seeded = true;
        dist
    }

    fn finish(&mut self) {
        if self.seeded {
            self.axes.fill(Dist::ZERO);
            self.seeded = false;
        }
    }

    #[inline(always)]
    fn axis(&self, dim: usize) -> Dist {
        self.axes[dim]
    }

    #[inline(always)]
    #[must_use = "the previous axis contribution has to be restored by `leave`"]
    fn enter(&mut self, dim: usize, axis: Dist) -> Dist {
        std::mem::replace(&mut self.axes[dim], axis)
    }

    #[inline(always)]
    fn leave(&mut self, dim: usize, saved: Dist) {
        self.axes[dim] = saved;
    }
}

struct Descent<'a, K: Strategy> {
    tree: &'a Tree,
    /// Hoisted from [`Tree::n_points`] so the division is not redone per query.
    n_points: usize,
    params: QueryParams,
    q: &'a [f64],
    strategy: PhantomData<K>,
}

impl<K: Strategy> Descent<'_, K> {
    fn run<B: Best>(&self, s: &mut Scratch<B>, out_d: &mut [f64], out_i: &mut [i64]) {
        let metric = self.params.metric;
        s.best.reset();
        let seed = s.cell.start(metric, self.q, self.tree.root_box());
        if self.admits(seed, s) {
            self.descend(ROOT, seed, s);
        }
        s.cell.finish();
        s.best.write_results(out_d, out_i, self.n_points, metric);
    }

    #[inline(always)]
    fn admits<B: Best>(&self, dist: Dist, s: &Scratch<B>) -> bool {
        self.admits_at(dist, s.best.bound(self.params.cutoff))
    }

    /// [`Descent::admits`] with the bound already in hand, for sites that hoist
    /// one `bound` across several tests.
    #[inline(always)]
    fn admits_at(&self, dist: Dist, bound: Dist) -> bool {
        dist.get() * self.params.eps_factor <= bound.get()
    }

    /// `cell_dist` is the lower bound of `node_id`'s cell; its per-axis parts are
    /// in `s.cell`.
    fn descend<B: Best>(&self, node_id: u32, cell_dist: Dist, s: &mut Scratch<B>) {
        let QueryParams { cutoff, metric, .. } = self.params;

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

                let far_axis = metric.reduce(diff.abs());
                let far_dist = metric.replace_axis(cell_dist, s.cell.axis(dim), far_axis);

                if order_by_box {
                    self.descend_by_box(near, far, dim, cell_dist, far_axis, far_dist, s);
                    return;
                }

                self.descend(near, cell_dist, s);

                let bound = s.best.bound(cutoff);
                if self.admits_at(far_dist, bound) {
                    let far_box = self.tree.box_of(far);
                    if self.admits_at(kernel::box_dist(metric, self.q, far_box), bound) {
                        self.enter_far(far, dim, far_axis, far_dist, s);
                    }
                }
            }
        }
    }

    /// Visiting one child tightens the bound for the other, so both box gates are
    /// tested as we go. Each child still descends on its plane bound — see the
    /// module invariant.
    ///
    /// Out of line so flat data, which never sets the flag, pays only the test.
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn descend_by_box<B: Best>(
        &self,
        near: u32,
        far: u32,
        dim: usize,
        cell_dist: Dist,
        far_axis: Dist,
        far_dist: Dist,
        s: &mut Scratch<B>,
    ) {
        let metric = self.params.metric;
        let near_box = kernel::box_dist(metric, self.q, self.tree.box_of(near));
        let far_box = kernel::box_dist(metric, self.q, self.tree.box_of(far));

        if near_box <= far_box {
            if self.admits(near_box, s) {
                self.descend(near, cell_dist, s);
            }
            if self.admits(far_box, s) {
                self.enter_far(far, dim, far_axis, far_dist, s);
            }
        } else {
            if self.admits(far_box, s) {
                self.enter_far(far, dim, far_axis, far_dist, s);
            }
            if self.admits(near_box, s) {
                self.descend(near, cell_dist, s);
            }
        }
    }

    #[inline(always)]
    fn enter_far<B: Best>(
        &self,
        far: u32,
        dim: usize,
        far_axis: Dist,
        far_dist: Dist,
        s: &mut Scratch<B>,
    ) {
        let saved = s.cell.enter(dim, far_axis);
        self.descend(far, far_dist, s);
        s.cell.leave(dim, saved);
    }

    #[inline]
    fn scan_leaf<B: Best>(&self, start: usize, end: usize, s: &mut Scratch<B>) {
        let QueryParams { cutoff, metric, .. } = self.params;
        let mut sink = LeafSink {
            best: &mut s.best,
            originals: &self.tree.indices,
            start,
            cutoff,
        };
        K::scan(metric, self.q, self.tree.rows(start, end), &mut sink);
    }
}

/// Leaf-relative offsets in, original row indices out.
struct LeafSink<'a, B> {
    best: &'a mut B,
    originals: &'a [u32],
    start: usize,
    cutoff: Dist,
}

impl<B: Best> kernel::Sink for LeafSink<'_, B> {
    #[inline(always)]
    fn bound(&self) -> Dist {
        self.best.bound(self.cutoff)
    }

    #[inline(always)]
    fn offer(&mut self, offset: usize, dist: Dist) {
        self.best
            .consider(dist, self.originals[self.start + offset]);
    }
}

/// The descent is monomorphized over this so the ubiquitous `k == 1` case lives
/// in two registers instead of heap buffers.
trait Best {
    fn new(k: usize) -> Self;
    fn reset(&mut self);
    /// The k-th best reduced distance so far, capped at `cutoff`.
    fn bound(&self, cutoff: Dist) -> Dist;
    /// Ties must resolve toward the smaller original index: SciPy is the oracle
    /// in `tests/test_kdtree.py`, and indices are compared exactly.
    fn consider(&mut self, dist: Dist, idx: u32);
    fn write_results(&self, out_d: &mut [f64], out_i: &mut [i64], n_points: usize, m: Metric);
}

struct Best1 {
    dist: Dist,
    i: u32,
}

impl Best for Best1 {
    fn new(_k: usize) -> Self {
        Self {
            dist: Dist::INFINITY,
            i: u32::MAX,
        }
    }

    fn reset(&mut self) {
        self.dist = Dist::INFINITY;
        self.i = u32::MAX;
    }

    #[inline]
    fn bound(&self, cutoff: Dist) -> Dist {
        self.dist.min(cutoff)
    }

    #[inline]
    fn consider(&mut self, dist: Dist, idx: u32) {
        if dist < self.dist || (dist == self.dist && idx < self.i) {
            self.dist = dist;
            self.i = idx;
        }
    }

    fn write_results(&self, out_d: &mut [f64], out_i: &mut [i64], n_points: usize, m: Metric) {
        if self.i != u32::MAX {
            out_d[0] = m.restore(self.dist);
            out_i[0] = self.i as i64;
        } else {
            out_d[0] = f64::INFINITY;
            out_i[0] = n_points as i64;
        }
    }
}

struct BestK {
    k: usize,
    dists: Vec<Dist>,
    ids: Vec<u32>,
}

impl Best for BestK {
    fn new(k: usize) -> Self {
        Self {
            k,
            dists: Vec::with_capacity(k),
            ids: Vec::with_capacity(k),
        }
    }

    fn reset(&mut self) {
        self.dists.clear();
        self.ids.clear();
    }

    #[inline]
    fn bound(&self, cutoff: Dist) -> Dist {
        if self.dists.len() < self.k {
            cutoff
        } else {
            self.dists[self.k - 1].min(cutoff)
        }
    }

    #[inline]
    fn consider(&mut self, dist: Dist, idx: u32) {
        if self.dists.len() == self.k {
            let worst_dist = self.dists[self.k - 1];
            if dist > worst_dist || (dist == worst_dist && self.ids[self.k - 1] <= idx) {
                return;
            }
            self.dists.pop();
            self.ids.pop();
        }
        let mut pos = self.dists.len();
        while pos > 0 {
            let prev_dist = self.dists[pos - 1];
            if prev_dist < dist || (prev_dist == dist && self.ids[pos - 1] < idx) {
                break;
            }
            pos -= 1;
        }
        self.dists.insert(pos, dist);
        self.ids.insert(pos, idx);
    }

    fn write_results(&self, out_d: &mut [f64], out_i: &mut [i64], n_points: usize, m: Metric) {
        for j in 0..out_d.len() {
            if j < self.dists.len() {
                out_d[j] = m.restore(self.dists[j]);
                out_i[j] = self.ids[j] as i64;
            } else {
                out_d[j] = f64::INFINITY;
                out_i[j] = n_points as i64;
            }
        }
    }
}
