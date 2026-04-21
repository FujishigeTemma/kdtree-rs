use std::cmp::Ordering;
use std::collections::BinaryHeap;

use rayon::prelude::*;

use crate::error::KDTreeError;
use crate::metric::Metric;
use crate::tree::Tree;

#[derive(Debug, Clone, Copy)]
struct Neighbor {
    distance_accum: f64,
    index: usize,
}

impl PartialEq for Neighbor {
    fn eq(&self, other: &Self) -> bool {
        self.distance_accum == other.distance_accum && self.index == other.index
    }
}

impl Eq for Neighbor {}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance_accum
            .total_cmp(&other.distance_accum)
            .then_with(|| self.index.cmp(&other.index))
    }
}

impl Tree {
    pub fn query_one(
        &self,
        query: &[f64],
        k: usize,
        p: f64,
        max_distance: Option<f64>,
        eps: f64,
    ) -> Result<(Vec<f64>, Vec<usize>), KDTreeError> {
        if query.len() != self.ndim() {
            return Err(KDTreeError::DimensionMismatch {
                expected: self.ndim(),
                got: query.len(),
            });
        }
        if k == 0 {
            return Err(KDTreeError::InvalidK);
        }
        if !query.iter().all(|value| value.is_finite()) {
            return Err(KDTreeError::NonFiniteData);
        }
        if !eps.is_finite() || eps < 0.0 {
            return Err(KDTreeError::InvalidEps(eps));
        }

        let metric = Metric::new(p)?;
        let max_distance = max_distance.unwrap_or(f64::INFINITY);
        if !max_distance.is_infinite() && (!max_distance.is_finite() || max_distance < 0.0) {
            return Err(KDTreeError::InvalidMaxDistance(max_distance));
        }

        let limit = metric.to_accum(max_distance);
        let eps_factor = metric.eps_factor(eps);
        let mut heap = BinaryHeap::with_capacity(k);
        let mut stack = vec![self.root()];

        while let Some(node_index) = stack.pop() {
            let upper_bound = current_upper_bound(&heap, k, limit);
            let (mins, maxes) = self.bbox(node_index);
            if metric.bbox_point_lower_bound(query, mins, maxes) > upper_bound / eps_factor {
                continue;
            }

            let node = self.node(node_index);
            if node.leaf {
                for &point_index in &self.points_indexed()[node.start..node.end] {
                    let distance_accum = metric.point_accum(query, self.point(point_index));
                    if distance_accum > limit {
                        continue;
                    }
                    push_neighbor(&mut heap, k, Neighbor {
                        distance_accum,
                        index: point_index,
                    });
                }
                continue;
            }

            let split_dim = node.split_dim;
            let (near, far) = if query[split_dim] <= node.split_value {
                (node.left, node.right)
            } else {
                (node.right, node.left)
            };
            if let Some(far) = far {
                stack.push(far);
            }
            if let Some(near) = near {
                stack.push(near);
            }
        }

        Ok(finalize_neighbors(heap, k, self.n_points(), metric))
    }

    pub fn query_many(
        &self,
        queries: &[f64],
        n_queries: usize,
        k: usize,
        p: f64,
        max_distance: Option<f64>,
        eps: f64,
        parallel: bool,
    ) -> Result<(Vec<f64>, Vec<usize>), KDTreeError> {
        if queries.len() != n_queries * self.ndim() {
            return Err(KDTreeError::InvalidShape("queries must be a contiguous row-major matrix"));
        }
        if parallel && n_queries > 1 {
            let results = (0..n_queries)
                .into_par_iter()
                .map(|query_index| {
                    let start = query_index * self.ndim();
                    self.query_one(&queries[start..start + self.ndim()], k, p, max_distance, eps)
                })
                .collect::<Vec<_>>();
            let mut distances = Vec::with_capacity(n_queries * k);
            let mut indices = Vec::with_capacity(n_queries * k);
            for result in results {
                let (dist, idx) = result?;
                distances.extend(dist);
                indices.extend(idx);
            }
            Ok((distances, indices))
        } else {
            let mut distances = Vec::with_capacity(n_queries * k);
            let mut indices = Vec::with_capacity(n_queries * k);
            for query_index in 0..n_queries {
                let start = query_index * self.ndim();
                let (dist, idx) =
                    self.query_one(&queries[start..start + self.ndim()], k, p, max_distance, eps)?;
                distances.extend(dist);
                indices.extend(idx);
            }
            Ok((distances, indices))
        }
    }
}

fn current_upper_bound(heap: &BinaryHeap<Neighbor>, k: usize, limit: f64) -> f64 {
    if heap.len() < k {
        limit
    } else {
        heap.peek().map(|neighbor| neighbor.distance_accum.min(limit)).unwrap_or(limit)
    }
}

fn push_neighbor(heap: &mut BinaryHeap<Neighbor>, k: usize, candidate: Neighbor) {
    if heap.len() < k {
        heap.push(candidate);
        return;
    }
    let replace = heap
        .peek()
        .map(|current| {
            candidate.distance_accum < current.distance_accum
                || (candidate.distance_accum == current.distance_accum && candidate.index < current.index)
        })
        .unwrap_or(true);
    if replace {
        let _ = heap.pop();
        heap.push(candidate);
    }
}

fn finalize_neighbors(
    heap: BinaryHeap<Neighbor>,
    k: usize,
    missing_index: usize,
    metric: Metric,
) -> (Vec<f64>, Vec<usize>) {
    let mut neighbors = heap.into_vec();
    neighbors.sort_by(|lhs, rhs| {
        lhs.distance_accum
            .total_cmp(&rhs.distance_accum)
            .then_with(|| lhs.index.cmp(&rhs.index))
    });
    let mut distances = neighbors
        .iter()
        .map(|neighbor| metric.finish(neighbor.distance_accum))
        .collect::<Vec<_>>();
    let mut indices = neighbors.iter().map(|neighbor| neighbor.index).collect::<Vec<_>>();
    while distances.len() < k {
        distances.push(f64::INFINITY);
        indices.push(missing_index);
    }
    (distances, indices)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::array;

    use crate::tree::Tree;

    #[test]
    fn query_one_returns_exact_nearest_neighbors() {
        let data = array![[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [5.0, 0.0]];
        let tree = Tree::new(data.view(), 2).expect("tree should build");

        let (distances, indices) = tree
            .query_one(&[1.5, 0.0], 2, 2.0, None, 0.0)
            .expect("query should succeed");

        assert_eq!(indices, vec![1, 0]);
        assert_relative_eq!(distances[0], 0.5);
        assert_relative_eq!(distances[1], 1.5);
    }

    #[test]
    fn query_many_pads_missing_neighbors() {
        let data = array![[0.0, 0.0], [10.0, 0.0]];
        let tree = Tree::new(data.view(), 1).expect("tree should build");

        let (distances, indices) = tree
            .query_many(&[0.0, 0.0, 11.0, 0.0], 2, 3, 2.0, Some(2.0), 0.0, false)
            .expect("query should succeed");

        assert_eq!(indices.len(), 6);
        assert_eq!(indices[2], 2);
        assert!(distances[2].is_infinite());
        assert_eq!(indices[5], 2);
        assert!(distances[5].is_infinite());
    }
}
