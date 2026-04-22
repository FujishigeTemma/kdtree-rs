use rayon::prelude::*;

use crate::error::KDTreeError;
use crate::metric::Metric;
use crate::node::Node;
use crate::tree::Tree;

impl Tree {
    pub fn query(
        &self,
        queries: &[f64],
        k: usize,
        p: f64,
        max_distance: Option<f64>,
        eps: f64,
        parallel: bool,
    ) -> Result<(Vec<f64>, Vec<usize>), KDTreeError> {
        let ndim = self.ndim();
        if k == 0 {
            return Err(KDTreeError::InvalidK);
        }
        if queries.is_empty() || queries.len() % ndim != 0 {
            return Err(KDTreeError::InvalidShape(
                "queries must be a contiguous row-major matrix",
            ));
        }
        if !queries.iter().all(|value| value.is_finite()) {
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
        let limit = metric.to_accum(max_distance);
        let eps_factor = metric.eps_factor(eps);
        let n_queries = queries.len() / ndim;
        let n_points = self.n_points();

        let mut distances = vec![0.0_f64; n_queries * k];
        let mut indices = vec![0_usize; n_queries * k];

        let run = |q_idx: usize, out_d: &mut [f64], out_i: &mut [usize]| {
            let mut scratch = QueryScratch::new(ndim, k);
            let q = &queries[q_idx * ndim..(q_idx + 1) * ndim];
            scratch.seed_from_root(q, self.root_bbox(), metric);
            self.descend(self.root(), q, k, limit, eps_factor, metric, &mut scratch);
            scratch.write_results(out_d, out_i, k, n_points, metric);
        };

        if parallel && n_queries > 1 {
            distances
                .par_chunks_mut(k)
                .zip(indices.par_chunks_mut(k))
                .enumerate()
                .for_each(|(q_idx, (d_chunk, i_chunk))| run(q_idx, d_chunk, i_chunk));
        } else {
            distances
                .chunks_mut(k)
                .zip(indices.chunks_mut(k))
                .enumerate()
                .for_each(|(q_idx, (d_chunk, i_chunk))| run(q_idx, d_chunk, i_chunk));
        }

        Ok((distances, indices))
    }

    /// Recursive branch-and-bound descent. `min_dist` is the L^p-accumulated
    /// distance from `q` to `node`'s split-cell bounding box. Per-axis
    /// contributions live in `scratch.side` and are mutated incrementally on
    /// the descent into the far child, then restored on return.
    fn descend(
        &self,
        node_id: u32,
        q: &[f64],
        k: usize,
        limit: f64,
        eps_factor: f64,
        metric: Metric,
        scratch: &mut QueryScratch,
    ) {
        let upper = scratch.upper(k, limit);
        if scratch.min_dist * eps_factor > upper {
            return;
        }

        match *self.node(node_id) {
            Node::Leaf { start, end } => {
                self.scan_leaf(start as usize, end as usize, q, k, limit, metric, scratch);
            }
            Node::Inner {
                left,
                right,
                split_dim,
                split_value,
            } => {
                let dim = split_dim as usize;
                let diff = q[dim] - split_value;
                let (near, far) = if diff <= 0.0 { (left, right) } else { (right, left) };

                self.descend(near, q, k, limit, eps_factor, metric, scratch);

                let new_axis = metric.axis_accum(diff.abs());
                let old_axis = scratch.side[dim];
                let new_min = metric.replace_axis(scratch.min_dist, old_axis, new_axis);
                let upper = scratch.upper(k, limit);
                if new_min * eps_factor <= upper {
                    let saved_min = scratch.min_dist;
                    scratch.side[dim] = new_axis;
                    scratch.min_dist = new_min;
                    self.descend(far, q, k, limit, eps_factor, metric, scratch);
                    scratch.side[dim] = old_axis;
                    scratch.min_dist = saved_min;
                }
            }
        }
    }

    #[inline]
    fn scan_leaf(
        &self,
        start: usize,
        end: usize,
        q: &[f64],
        k: usize,
        limit: f64,
        metric: Metric,
        scratch: &mut QueryScratch,
    ) {
        let block = self.leaf_block(start, end);
        let originals = self.points_indexed();
        let ndim = self.ndim();
        for (offset, coords) in block.chunks_exact(ndim).enumerate() {
            let bound = scratch.upper(k, limit);
            let d = metric.point_accum(q, coords, bound);
            if d > bound {
                continue;
            }
            let idx = originals[start + offset];
            scratch.consider(d, idx, k);
        }
    }
}

/// Per-query mutable state. Holds the k-best so far, the per-axis distance
/// contribution of the current cell, and the current cell's accumulated
/// L^p lower bound.
struct QueryScratch {
    nb_d: Vec<f64>,
    nb_i: Vec<usize>,
    side: Vec<f64>,
    min_dist: f64,
}

impl QueryScratch {
    fn new(ndim: usize, k: usize) -> Self {
        Self {
            nb_d: Vec::with_capacity(k),
            nb_i: Vec::with_capacity(k),
            side: vec![0.0; ndim],
            min_dist: 0.0,
        }
    }

    fn seed_from_root(&mut self, q: &[f64], bbox: (&[f64], &[f64]), metric: Metric) {
        let (lo, hi) = bbox;
        let mut acc = 0.0_f64;
        for d in 0..q.len() {
            let off = if q[d] < lo[d] {
                lo[d] - q[d]
            } else if q[d] > hi[d] {
                q[d] - hi[d]
            } else {
                0.0
            };
            let axis = metric.axis_accum(off);
            self.side[d] = axis;
            acc = metric.fold_axis(acc, axis);
        }
        self.min_dist = acc;
    }

    #[inline]
    fn upper(&self, k: usize, limit: f64) -> f64 {
        if self.nb_d.len() < k {
            limit
        } else {
            self.nb_d[k - 1].min(limit)
        }
    }

    /// Insert `(d, idx)` into the sorted k-best buffer. Ties are resolved by
    /// smaller original index, matching `numpy.argsort(kind="stable")`.
    #[inline]
    fn consider(&mut self, d: f64, idx: usize, k: usize) {
        if self.nb_d.len() == k {
            let worst = self.nb_d[k - 1];
            if d > worst || (d == worst && self.nb_i[k - 1] <= idx) {
                return;
            }
        }
        let mut pos = self.nb_d.len().min(k);
        while pos > 0 {
            let prev_d = self.nb_d[pos - 1];
            if prev_d < d || (prev_d == d && self.nb_i[pos - 1] < idx) {
                break;
            }
            pos -= 1;
        }
        if self.nb_d.len() < k {
            self.nb_d.insert(pos, d);
            self.nb_i.insert(pos, idx);
        } else {
            for j in (pos + 1..k).rev() {
                self.nb_d[j] = self.nb_d[j - 1];
                self.nb_i[j] = self.nb_i[j - 1];
            }
            self.nb_d[pos] = d;
            self.nb_i[pos] = idx;
        }
    }

    fn write_results(
        &self,
        out_d: &mut [f64],
        out_i: &mut [usize],
        k: usize,
        n_points: usize,
        metric: Metric,
    ) {
        for j in 0..k {
            if j < self.nb_d.len() {
                out_d[j] = metric.finish(self.nb_d[j]);
                out_i[j] = self.nb_i[j];
            } else {
                out_d[j] = f64::INFINITY;
                out_i[j] = n_points;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::array;

    use crate::tree::Tree;

    #[test]
    fn query_returns_exact_nearest_neighbors() {
        let data = array![[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [5.0, 0.0]];
        let tree = Tree::new(data.view(), 2).expect("tree should build");

        let (distances, indices) = tree
            .query(&[1.5, 0.0], 2, 2.0, None, 0.0, false)
            .expect("query should succeed");

        assert_eq!(indices, vec![1, 0]);
        assert_relative_eq!(distances[0], 0.5);
        assert_relative_eq!(distances[1], 1.5);
    }

    #[test]
    fn query_pads_missing_neighbors() {
        let data = array![[0.0, 0.0], [10.0, 0.0]];
        let tree = Tree::new(data.view(), 1).expect("tree should build");

        let (distances, indices) = tree
            .query(&[0.0, 0.0, 11.0, 0.0], 3, 2.0, Some(2.0), 0.0, false)
            .expect("query should succeed");

        assert_eq!(indices.len(), 6);
        assert_eq!(indices[2], 2);
        assert!(distances[2].is_infinite());
        assert_eq!(indices[5], 2);
        assert!(distances[5].is_infinite());
    }
}
