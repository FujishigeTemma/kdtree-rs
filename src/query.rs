//! The descent prunes with two bounds, checked in order of cost: an
//! O(1)-updatable lower bound on the distance from the query to the current cell
//! (per-axis parts in [`CellBound`], the folded total passed down the
//! recursion), and — only when that fails to prune — the tight box of the points
//! the subtree actually contains. The tight box is what collapses degenerate
//! clustered data, where every split lands on the same few dimensions and plane
//! bounds stay uselessly small; the plane bound keeps the common well-separated
//! descent free of O(ndim) box work.
//!
//! **Box distances gate visits and nothing else.** A child is always entered
//! with a bound derived from [`CellBound`], never with its own box distance.
//! Feed a box distance into the recursion and it drifts out of step with the
//! per-axis breakdown, so every later `replace_axis` subtracts a contribution
//! that is not in the total — silently wrong results, not a crash.

use std::marker::PhantomData;

use rayon::prelude::*;

use crate::error::KDTreeError;
use crate::kernel::{self, Packed, Streamed, Strategy};
use crate::layout::{BBox, axis_offset};
use crate::metric::{Metric, Rd};
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

        // Monomorphizing over the metric as well was measured slower: the
        // const-folded kernels inline into `descend` and bloat it, where the
        // enum dispatch stays perfectly predicted.
        let args = (queries, k, params, parallel);
        match (k == 1, kernel::packs(metric, self.ndim())) {
            (true, true) => self.run::<Best1, Packed>(args, &mut distances, &mut indices),
            (true, false) => self.run::<Best1, Streamed>(args, &mut distances, &mut indices),
            (false, true) => self.run::<BestK, Packed>(args, &mut distances, &mut indices),
            (false, false) => self.run::<BestK, Streamed>(args, &mut distances, &mut indices),
        }

        Ok((distances, indices))
    }

    fn run<B: Best, K: Strategy>(
        &self,
        (queries, k, params, parallel): (&[f64], usize, QueryParams, bool),
        distances: &mut [f64],
        indices: &mut [i64],
    ) {
        let ndim = self.ndim();
        let n_queries = queries.len() / ndim;
        let run = |scratch: &mut Scratch<B>, i: usize, out_d: &mut [f64], out_i: &mut [i64]| {
            let descent = Descent::<K> {
                tree: self,
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
    /// `max_distance`, reduced.
    cutoff: Rd,
    eps_factor: f64,
    metric: Metric,
}

/// Allocated once per thread and reused across queries. Kept separate from
/// [`Descent`] so the recursion borrows read-only context and mutable state
/// independently — one `&mut` over both forces a context reload after every
/// recursive call.
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
/// breakdown that makes `replace_axis` an O(1) update. The folded total is not
/// here: it is per-frame and travels as a recursion argument, since shared
/// mutable state would force a reload after every recursive call.
///
/// Invariant: all axes are zero between queries. That is already the answer for
/// a query inside the root box, which is the overwhelmingly common case, so
/// [`CellBound::start`] normally writes nothing and [`CellBound::finish`] has
/// nothing to undo.
struct CellBound {
    axes: Vec<Rd>,
    seeded: bool,
}

impl CellBound {
    fn new(ndim: usize) -> Self {
        Self {
            axes: vec![Rd::ZERO; ndim],
            seeded: false,
        }
    }

    /// One vectorized box distance decides the common case, and that decision is
    /// what inlines into the caller; the rare fill stays out of line.
    #[inline(always)]
    fn start(&mut self, m: Metric, q: &[f64], root: BBox<'_>) -> Rd {
        if kernel::box_rd(m, q, root) == Rd::ZERO {
            return Rd::ZERO;
        }
        self.seed(m, q, root)
    }

    fn seed(&mut self, m: Metric, q: &[f64], root: BBox<'_>) -> Rd {
        let mut rd = Rd::ZERO;
        let bounds = q.iter().zip(root.lo).zip(root.hi);
        for (slot, ((&q, &lo), &hi)) in self.axes.iter_mut().zip(bounds) {
            let axis = m.reduce(axis_offset(q, lo, hi));
            *slot = axis;
            rd = m.fold(rd, axis);
        }
        self.seeded = true;
        rd
    }

    fn finish(&mut self) {
        if self.seeded {
            self.axes.fill(Rd::ZERO);
            self.seeded = false;
        }
    }

    #[inline(always)]
    fn axis(&self, dim: usize) -> Rd {
        self.axes[dim]
    }

    #[inline(always)]
    #[must_use = "the previous axis contribution has to be restored by `leave`"]
    fn enter(&mut self, dim: usize, axis: Rd) -> Rd {
        std::mem::replace(&mut self.axes[dim], axis)
    }

    #[inline(always)]
    fn leave(&mut self, dim: usize, saved: Rd) {
        self.axes[dim] = saved;
    }
}

struct Descent<'a, K: Strategy> {
    tree: &'a Tree,
    params: QueryParams,
    q: &'a [f64],
    strategy: PhantomData<K>,
}

impl<K: Strategy> Descent<'_, K> {
    fn run<B: Best>(&self, s: &mut Scratch<B>, out_d: &mut [f64], out_i: &mut [i64]) {
        let QueryParams {
            cutoff,
            eps_factor,
            metric,
        } = self.params;
        s.best.reset();
        let seed = s.cell.start(metric, self.q, self.tree.root_box());
        if seed.scaled(eps_factor) <= cutoff {
            self.descend(ROOT, seed, s);
        }
        s.cell.finish();
        s.best
            .write_results(out_d, out_i, self.tree.n_points(), metric);
    }

    /// The approximation factor scales the lower bound rather than the k-best
    /// bound, so the comparison stays in the reduced domain.
    #[inline(always)]
    fn admits<B: Best>(&self, rd: Rd, s: &Scratch<B>) -> bool {
        rd.scaled(self.params.eps_factor) <= s.best.bound(self.params.cutoff)
    }

    /// `cell_rd` is the lower bound of `node_id`'s cell; its per-axis parts are
    /// in `s.cell`.
    fn descend<B: Best>(&self, node_id: u32, cell_rd: Rd, s: &mut Scratch<B>) {
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

                let far_axis = metric.reduce(diff.abs());
                let far_rd = metric.replace_axis(cell_rd, s.cell.axis(dim), far_axis);

                if order_by_box {
                    self.descend_by_box(near, far, dim, cell_rd, far_axis, far_rd, s);
                    return;
                }

                self.descend(near, cell_rd, s);

                let bound = s.best.bound(cutoff);
                if far_rd.scaled(eps_factor) <= bound {
                    let far_box = self.tree.box_of(far);
                    if kernel::box_rd(metric, self.q, far_box).scaled(eps_factor) <= bound {
                        self.enter_far(far, dim, far_axis, far_rd, s);
                    }
                }
            }
        }
    }

    /// Visiting one child tightens the bound for the other, so both box gates are
    /// tested as we go. Each child still descends on its plane bound (`cell_rd` /
    /// `far_rd`) — see the module invariant.
    ///
    /// Out of line so that flat data, which never sets the flag, pays only the
    /// flag test.
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn descend_by_box<B: Best>(
        &self,
        near: u32,
        far: u32,
        dim: usize,
        cell_rd: Rd,
        far_axis: Rd,
        far_rd: Rd,
        s: &mut Scratch<B>,
    ) {
        let metric = self.params.metric;
        let near_box = kernel::box_rd(metric, self.q, self.tree.box_of(near));
        let far_box = kernel::box_rd(metric, self.q, self.tree.box_of(far));

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

    #[inline(always)]
    fn enter_far<B: Best>(&self, far: u32, dim: usize, far_axis: Rd, far_rd: Rd, s: &mut Scratch<B>) {
        let saved = s.cell.enter(dim, far_axis);
        self.descend(far, far_rd, s);
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
    cutoff: Rd,
}

impl<B: Best> kernel::Sink for LeafSink<'_, B> {
    #[inline(always)]
    fn bound(&self) -> Rd {
        self.best.bound(self.cutoff)
    }

    #[inline(always)]
    fn offer(&mut self, offset: usize, rd: Rd) {
        self.best.consider(rd, self.originals[self.start + offset]);
    }
}

/// The descent is monomorphized over this so the ubiquitous `k == 1` case lives
/// in two registers instead of heap buffers.
trait Best {
    fn new(k: usize) -> Self;
    fn reset(&mut self);
    /// The k-th best reduced distance so far, capped at `cutoff`.
    fn bound(&self, cutoff: Rd) -> Rd;
    /// Ties must resolve toward the smaller original index: SciPy is the oracle
    /// in `tests/test_kdtree.py`, and indices are compared exactly.
    fn consider(&mut self, rd: Rd, idx: u32);
    fn write_results(&self, out_d: &mut [f64], out_i: &mut [i64], n_points: usize, m: Metric);
}

struct Best1 {
    rd: Rd,
    i: u32,
}

impl Best for Best1 {
    fn new(_k: usize) -> Self {
        Self {
            rd: Rd::INFINITY,
            i: u32::MAX,
        }
    }

    fn reset(&mut self) {
        self.rd = Rd::INFINITY;
        self.i = u32::MAX;
    }

    #[inline]
    fn bound(&self, cutoff: Rd) -> Rd {
        self.rd.min(cutoff)
    }

    #[inline]
    fn consider(&mut self, rd: Rd, idx: u32) {
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

/// A sorted buffer, insertion-sorted on the fly.
struct BestK {
    k: usize,
    rds: Vec<Rd>,
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
    fn bound(&self, cutoff: Rd) -> Rd {
        if self.rds.len() < self.k {
            cutoff
        } else {
            self.rds[self.k - 1].min(cutoff)
        }
    }

    #[inline]
    fn consider(&mut self, rd: Rd, idx: u32) {
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
