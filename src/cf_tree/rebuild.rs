//! CF-Tree rebuild and threshold estimation.

use crate::cluster_feature::ClusterFeature;
use crate::distance::CFDistance;
use crate::types::Float;

use super::{CFNode, CFTree};

impl<F: Float, CF: ClusterFeature<F>, D: CFDistance<F, CF>, A: CFDistance<F, CF>>
  CFTree<F, CF, D, A>
{
  pub fn estimate_threshold(&self) -> F {
    // Optimization: accumulate sum and count directly instead of collecting into Vec
    let (sum, count) = self
      .nodes
      .iter()
      .fold((F::zero(), 0usize), |(mut sum, mut count), n| {
        if n.is_leaf() && n.children().len() > 1 {
          let children: Vec<&CF> = n
            .children()
            .iter()
            .map(|&idx| &self.leaf_entries[idx])
            .collect();

          let child_count = children.len();
          let mut best_dist = vec![F::infinity(); child_count];
          let mut best_idx = vec![0usize; child_count];

          // We compute distances twice (Phase 1 with dist_function,
          // Phase 2 with abs_function). This allows different distance metrics
          // for tree routing vs. absorption.
          for i in 0..child_count {
            for j in (i + 1)..child_count {
              let d = self
                .dist_function
                .sq_dist_cf(children[i], children[j], self.dim);
              if d < best_dist[i] {
                best_dist[i] = d;
                best_idx[i] = j;
              }
              if d < best_dist[j] {
                best_dist[j] = d;
                best_idx[j] = i;
              }
            }
          }

          // Phase 2: accumulate threshold values using abs_function.
          for i in 0..child_count {
            let threshold =
              self
                .abs_function
                .sq_dist_cf(children[i], children[best_idx[i]], self.dim);
            sum += threshold.sqrt();
            count += 1;
          }
        }
        (sum, count)
      });

    if count == 0 {
      F::zero()
    } else {
      // sqrt() is kept for all distance measures to match ELKI's threshold estimation,
      // even though it's unnecessary for non-squared distances like Manhattan.
      let mean = sum / F::from_index(count);
      // (1 + 4*eps) guards against sqrt(D)^2 < D rounding at the boundary.
      mean * mean * (F::one() + F::from_index(4) * F::epsilon())
    }
  }

  fn collect_leaf_entries_in_tree_order(&self) -> Vec<CF> {
    let mut entries = Vec::with_capacity(self.leaf_entries.len());
    self.collect_leaf_entries_from_node(self.root, &mut entries);
    debug_assert_eq!(entries.len(), self.leaf_entries.len());
    entries
  }

  fn collect_leaf_entries_from_node(&self, node_id: usize, out: &mut Vec<CF>) {
    let node = &self.nodes[node_id];
    if node.is_leaf() {
      for &entry_id in node.children() {
        out.push(self.leaf_entries[entry_id].clone());
      }
      return;
    }
    for &child_id in node.children() {
      self.collect_leaf_entries_from_node(child_id, out);
    }
  }

  #[cfg(test)]
  fn collect_leaf_entry_ids_in_tree_order(&self) -> Vec<usize> {
    let mut ids = Vec::with_capacity(self.leaf_entries.len());
    self.collect_leaf_entry_ids_from_node(self.root, &mut ids);
    ids
  }

  #[cfg(test)]
  fn collect_leaf_entry_ids_from_node(&self, node_id: usize, out: &mut Vec<usize>) {
    let node = &self.nodes[node_id];
    if node.is_leaf() {
      out.extend(node.children().iter().copied());
      return;
    }
    for &child_id in node.children() {
      self.collect_leaf_entry_ids_from_node(child_id, out);
    }
  }

  pub fn rebuild_tree(&mut self) {
    self.rebuild_count += 1;
    let threshold = self.estimate_threshold();
    // Use DFS leaf order (not storage order) to match Java's rebuild behavior.
    // After splits, storage order diverges from tree order, producing different
    // tree shapes and cluster quality.
    let leaf_entries = self.collect_leaf_entries_in_tree_order();
    // Always raise the threshold; never lower it.
    self.threshold = if self.threshold < threshold {
      threshold
    } else {
      self.threshold
    };
    self.leaf_entries.clear();
    self.root = 0;
    self.nodes.clear();
    self.nodes.push(CFNode::new(self.dim, self.capacity, true));
    // Reinsert in reverse tree order to mirror Java's rebuild loop.
    let n_entries = leaf_entries.len();
    for leaf in leaf_entries.into_iter().rev() {
      unsafe {
        self.insert_unchecked_cf(leaf);
      }
    }
    assert!(self.leaf_entries.len() <= n_entries);
  }
}

// ── Tests ──

#[cfg(test)]
mod tests {
  use crate::utils;
  use crate::{cf_tree::CFTree, distance::CentroidEuclideanDistance};

  fn create_data(size: usize, dim: usize) -> Vec<Vec<f64>> {
    use rand::rand_core::SeedableRng;
    use rand::{Rng, rngs::StdRng};

    let mut data = Vec::with_capacity(size);
    let mut rng = StdRng::seed_from_u64(0xBAD5EEDu64).random_iter();
    for _ in 0..size {
      data.push(rng.by_ref().take(dim).collect::<Vec<f64>>());
    }
    data
  }

  #[test]
  fn flat_tree_multiple_objects_per_leaf() {
    let data = create_data(50, 2);
    let mut tree = CFTree::new(
      CentroidEuclideanDistance::new(),
      CentroidEuclideanDistance::new(),
      100,
      2,
      5,
      0.,
    );
    for v in data.iter() {
      tree.insert(v);
    }
    assert!(tree.rebuild_count() > 0);
    assert_ne!(tree.threshold, 0.);
    assert_eq!(tree.leaf_entries().len(), 4);
    utils::verify_size(&data, tree.nodes().get(tree.root()).unwrap().as_cf());
    println!(
      "{}",
      utils::calc_error_for_mean(&data, tree.nodes().get(tree.root()).unwrap().as_cf())
    );
    utils::verify_node(&tree.nodes()[tree.root()], &tree);
  }

  #[test]
  fn test_rebuild_must_use_tree_leaf_order_not_storage_order() {
    // Java rebuilds from DFS leaf order, not from the leaf_entries backing
    // vector order. After enough splits these orders diverge.
    let data = create_data(40, 2);
    let mut tree = CFTree::new(
      CentroidEuclideanDistance::<f64, crate::cluster_feature::VII<f64>>::new(),
      CentroidEuclideanDistance::<f64, crate::cluster_feature::VII<f64>>::new(),
      3,
      2,
      100,
      0.0,
    );

    for v in data.iter() {
      tree.insert(v);
    }

    let storage_ids: Vec<usize> = (0..tree.leaf_entries().len()).collect();
    let dfs_ids = tree.collect_leaf_entry_ids_in_tree_order();

    let mut sorted_dfs = dfs_ids.clone();
    sorted_dfs.sort_unstable();
    assert_eq!(
      sorted_dfs, storage_ids,
      "DFS traversal must visit every leaf entry exactly once"
    );
    assert_ne!(
      dfs_ids, storage_ids,
      "leaf_entries storage order unexpectedly matched DFS order; rebuild regression test no longer exercises the Java mismatch"
    );
  }
}
