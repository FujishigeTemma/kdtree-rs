use rayon::prelude::*;

use crate::error::KDTreeError;
use crate::metric::Metric;
use crate::tree::Tree;

impl Tree {
    pub fn query_radius_one(
        &self,
        query: &[f64],
        radius: f64,
        p: f64,
        sort: bool,
    ) -> Result<(Vec<usize>, Vec<f64>), KDTreeError> {
        if query.len() != self.ndim() {
            return Err(KDTreeError::DimensionMismatch {
                expected: self.ndim(),
                got: query.len(),
            });
        }
        if !query.iter().all(|value| value.is_finite()) {
            return Err(KDTreeError::NonFiniteData);
        }
        if !radius.is_finite() || radius < 0.0 {
            return Err(KDTreeError::InvalidRadius(radius));
        }

        let metric = Metric::new(p)?;
        let radius_accum = metric.to_accum(radius);
        let mut stack = vec![self.root()];
        let mut pairs = Vec::new();

        while let Some(node_index) = stack.pop() {
            let (mins, maxes) = self.bbox(node_index);
            if metric.bbox_point_lower_bound(query, mins, maxes) > radius_accum {
                continue;
            }

            let node = self.node(node_index);
            if node.leaf {
                for &point_index in &self.points_indexed()[node.start..node.end] {
                    let accum = metric.point_accum(query, self.point(point_index));
                    if accum <= radius_accum {
                        pairs.push((point_index, metric.finish(accum)));
                    }
                }
                continue;
            }

            if let Some(left) = node.left {
                stack.push(left);
            }
            if let Some(right) = node.right {
                stack.push(right);
            }
        }

        if sort {
            pairs.sort_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1).then_with(|| lhs.0.cmp(&rhs.0)));
        }
        let indices = pairs.iter().map(|(index, _)| *index).collect();
        let distances = pairs.iter().map(|(_, distance)| *distance).collect();
        Ok((indices, distances))
    }

    pub fn query_radius_many(
        &self,
        queries: &[f64],
        n_queries: usize,
        radius: f64,
        p: f64,
        sort: bool,
        parallel: bool,
    ) -> Result<(Vec<Vec<usize>>, Vec<Vec<f64>>), KDTreeError> {
        if queries.len() != n_queries * self.ndim() {
            return Err(KDTreeError::InvalidShape("queries must be a contiguous row-major matrix"));
        }

        let results = if parallel && n_queries > 1 {
            (0..n_queries)
                .into_par_iter()
                .map(|query_index| {
                    let start = query_index * self.ndim();
                    self.query_radius_one(&queries[start..start + self.ndim()], radius, p, sort)
                })
                .collect::<Vec<_>>()
        } else {
            (0..n_queries)
                .map(|query_index| {
                    let start = query_index * self.ndim();
                    self.query_radius_one(&queries[start..start + self.ndim()], radius, p, sort)
                })
                .collect::<Vec<_>>()
        };

        let mut indices = Vec::with_capacity(n_queries);
        let mut distances = Vec::with_capacity(n_queries);
        for result in results {
            let (query_indices, query_distances) = result?;
            indices.push(query_indices);
            distances.push(query_distances);
        }
        Ok((indices, distances))
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::array;

    use crate::tree::Tree;

    #[test]
    fn query_radius_finds_all_matches() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [4.0, 0.0]];
        let tree = Tree::new(data.view(), 2).expect("tree should build");

        let (indices, distances) = tree
            .query_radius_one(&[0.0, 0.0], 1.1, 2.0, true)
            .expect("radius query should succeed");

        assert_eq!(indices, vec![0, 1]);
        assert_relative_eq!(distances[0], 0.0);
        assert_relative_eq!(distances[1], 1.0);
    }
}
