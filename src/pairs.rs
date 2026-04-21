use crate::error::KDTreeError;
use crate::metric::Metric;
use crate::tree::Tree;

impl Tree {
    pub fn query_pairs(&self, radius: f64, p: f64) -> Result<Vec<[usize; 2]>, KDTreeError> {
        if !radius.is_finite() || radius < 0.0 {
            return Err(KDTreeError::InvalidRadius(radius));
        }
        let metric = Metric::new(p)?;
        let radius_accum = metric.to_accum(radius);
        let mut pairs = Vec::new();
        self.collect_pairs(self.root(), self.root(), true, metric, radius_accum, &mut pairs);
        pairs.sort_unstable();
        Ok(pairs)
    }

    fn collect_pairs(
        &self,
        left_node: usize,
        right_node: usize,
        same_branch: bool,
        metric: Metric,
        radius_accum: f64,
        output: &mut Vec<[usize; 2]>,
    ) {
        let (left_mins, left_maxes) = self.bbox(left_node);
        let (right_mins, right_maxes) = self.bbox(right_node);
        if metric.bbox_bbox_lower_bound(left_mins, left_maxes, right_mins, right_maxes) > radius_accum {
            return;
        }

        let left = self.node(left_node);
        let right = self.node(right_node);

        if left.leaf && right.leaf {
            let lhs = &self.points_indexed()[left.start..left.end];
            let rhs = &self.points_indexed()[right.start..right.end];
            if same_branch {
                for (offset, &lhs_index) in lhs.iter().enumerate() {
                    for &rhs_index in &lhs[offset + 1..] {
                        let accum = metric.point_accum(self.point(lhs_index), self.point(rhs_index));
                        if accum <= radius_accum {
                            output.push([lhs_index.min(rhs_index), lhs_index.max(rhs_index)]);
                        }
                    }
                }
            } else {
                for &lhs_index in lhs {
                    for &rhs_index in rhs {
                        let accum = metric.point_accum(self.point(lhs_index), self.point(rhs_index));
                        if accum <= radius_accum {
                            output.push([lhs_index.min(rhs_index), lhs_index.max(rhs_index)]);
                        }
                    }
                }
            }
            return;
        }

        if same_branch {
            if let Some(left_left) = left.left {
                self.collect_pairs(left_left, left_left, true, metric, radius_accum, output);
                if let Some(left_right) = left.right {
                    self.collect_pairs(left_left, left_right, false, metric, radius_accum, output);
                    self.collect_pairs(left_right, left_right, true, metric, radius_accum, output);
                }
            }
            return;
        }

        let left_len = left.end - left.start;
        let right_len = right.end - right.start;
        if left.leaf || (!right.leaf && right_len > left_len) {
            if let Some(right_left) = right.left {
                self.collect_pairs(left_node, right_left, false, metric, radius_accum, output);
            }
            if let Some(right_right) = right.right {
                self.collect_pairs(left_node, right_right, false, metric, radius_accum, output);
            }
        } else {
            if let Some(left_left) = left.left {
                self.collect_pairs(left_left, right_node, false, metric, radius_accum, output);
            }
            if let Some(left_right) = left.right {
                self.collect_pairs(left_right, right_node, false, metric, radius_accum, output);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use crate::tree::Tree;

    #[test]
    fn query_pairs_returns_unique_pairs() {
        let data = array![[0.0, 0.0], [0.5, 0.0], [2.0, 0.0], [2.2, 0.0]];
        let tree = Tree::new(data.view(), 2).expect("tree should build");

        let pairs = tree.query_pairs(0.6, 2.0).expect("pairs query should succeed");

        assert_eq!(pairs, vec![[0, 1], [2, 3]]);
    }
}
