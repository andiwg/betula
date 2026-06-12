//! CF-Tree split operations.

use crate::cluster_feature::ClusterFeature;
use crate::distance::CFDistance;
use crate::types::Float;

use super::{CFNode, CFTree};

impl<F: Float, CF: ClusterFeature<F>, D: CFDistance<F, CF>, A: CFDistance<F, CF>>
  CFTree<F, CF, D, A>
{
  pub(crate) fn split_node(
    &self,
    child_ids: &[usize],
    is_leaf: bool,
  ) -> (CFNode<F, CF>, CFNode<F, CF>) {
    let childs: Vec<&CF> = if is_leaf {
      child_ids.iter().map(|&id| &self.leaf_entries[id]).collect()
    } else {
      child_ids.iter().map(|&id| self.nodes[id].as_cf()).collect()
    };

    let assignment = self.split_cfs(&childs);

    let mut new_node_one: CFNode<F, CF> = CFNode::new(self.dim, self.capacity, is_leaf);
    let mut new_node_two: CFNode<F, CF> = CFNode::new(self.dim, self.capacity, is_leaf);

    child_ids
      .iter()
      .zip(childs)
      .zip(assignment)
      .for_each(|((&cf_id, child), assign)| {
        if assign {
          new_node_one.add_child(cf_id);
          new_node_one.as_mut_cf().add_cf(child);
        } else {
          new_node_two.add_child(cf_id);
          new_node_two.as_mut_cf().add_cf(child);
        }
      });
    (new_node_one, new_node_two)
  }

  fn split_cfs(&self, cfs: &[&CF]) -> Vec<bool> {
    let num_cfs = cfs.len();
    if num_cfs <= 1 {
      return vec![true; num_cfs];
    }

    // Build packed triangular matrix (upper triangle, excluding diagonal)
    // Trade-off: Uses O(n²/2) memory to avoid re-computing distances to seeds.
    // For capacity=64: ~16KB per split, negligible.
    // Index formula: for pair (i,j) where i < j:
    //   idx = i * (2*n - i - 1) / 2 + (j - i - 1)
    let tri_size = num_cfs * (num_cfs - 1) / 2;
    let mut tri_dist = vec![F::zero(); tri_size];

    let mut max_dist = F::zero();
    let mut seed1 = 0;
    let mut seed2 = 1;

    for i in 0..num_cfs {
      for j in (i + 1)..num_cfs {
        let idx = i * (2 * num_cfs - i - 1) / 2 + (j - i - 1);
        let dist = self.dist_function.sq_dist_cf(cfs[i], cfs[j], self.dim);
        tri_dist[idx] = dist;
        if dist > max_dist {
          max_dist = dist;
          seed1 = i;
          seed2 = j;
        }
      }
    }

    // Helper to look up distance in packed triangular matrix
    fn get_dist<F: Float>(tri: &[F], n: usize, i: usize, j: usize) -> F {
      if i == j {
        F::zero()
      } else if i < j {
        tri[i * (2 * n - i - 1) / 2 + (j - i - 1)]
      } else {
        tri[j * (2 * n - j - 1) / 2 + (i - j - 1)]
      }
    }

    // Tie-breaking: balance by assigning ties to the group with fewer entries.
    // Mirrors Java's `si <= sj` logic in CFTree.split().
    let mut si = 0usize;
    let mut sj = 0usize;
    (0..num_cfs)
      .map(|i| {
        let d1 = get_dist(&tri_dist, num_cfs, i, seed1);
        let d2 = get_dist(&tri_dist, num_cfs, i, seed2);
        let assign_to_one = if d1 < d2 {
          true
        } else if d1 > d2 {
          false
        } else {
          // Tie: assign to whichever group currently has fewer entries.
          si <= sj
        };
        if assign_to_one {
          si += 1;
        } else {
          sj += 1;
        }
        assign_to_one
      })
      .collect()
  }
}

// ── Tests ──

#[cfg(test)]
mod tests {
  use rand::rand_core::SeedableRng;
  use rand::{Rng, rngs::StdRng};

  use crate::cluster_feature::ClusterFeature;
  use crate::utils;
  use crate::{cf_tree::CFTree, distance::CentroidEuclideanDistance};

  #[test]
  fn test_split_propagation_cf_consistency() {
    // Scenario: leaf overflows, parent has capacity (gets add(x)),
    // then another insert causes parent to overflow and split,
    // then split result propagates to a higher-level parent with capacity.
    // This should expose the CF double-update bug.

    let mut rng = StdRng::seed_from_u64(42);
    let data: Vec<Vec<f64>> = (0..20)
      .map(|_| vec![rng.random_range(0.0..100.0), rng.random_range(0.0..100.0)])
      .collect();

    let mut tree = CFTree::new(
      CentroidEuclideanDistance::new(),
      CentroidEuclideanDistance::new(),
      3,   // capacity
      2,   // dim
      100, // maxleaves
      0.0, // threshold
    );

    for v in data.iter() {
      tree.insert(v);
    }

    // Verify root CF size matches data count
    utils::verify_size(&data, tree.nodes().get(tree.root()).unwrap().as_cf());

    // Verify all nodes' CFs match their children's CFs
    utils::verify_node(&tree.nodes()[tree.root()], &tree);

    // Also verify total size across all leaf entries matches
    let total_size: usize = tree.leaf_entries().iter().map(|cf| cf.size()).sum();
    assert_eq!(
      total_size,
      data.len(),
      "Total leaf entry sizes ({}) should match data count ({})",
      total_size,
      data.len()
    );
  }

  #[test]
  fn test_cf_double_update_bug() {
    // Manually construct the exact bug scenario:
    // 1. Insert points until a leaf overflows
    // 2. Parent has capacity -> parent.as_mut_cf().add(x) is called
    // 3. Insert another point that causes the parent to overflow
    // 4. Parent splits, split result propagates to root with capacity
    // 5. Check if root's CF matches sum of children's CFs

    let mut tree = CFTree::new(
      CentroidEuclideanDistance::<f64, crate::cluster_feature::VII<f64>>::new(),
      CentroidEuclideanDistance::<f64, crate::cluster_feature::VII<f64>>::new(),
      3,   // capacity
      2,   // dim
      100, // maxleaves
      0.0, // threshold
    );

    let points: Vec<Vec<f64>> = vec![
      vec![1.0, 1.0],
      vec![2.0, 2.0],
      vec![3.0, 3.0],
      vec![4.0, 4.0],
    ];
    for p in &points {
      tree.insert(p);
    }

    let more_points: Vec<Vec<f64>> = vec![
      vec![5.0, 5.0],
      vec![6.0, 6.0],
      vec![7.0, 7.0],
      vec![8.0, 8.0],
    ];
    for p in &more_points {
      tree.insert(p);
    }

    let even_more: Vec<Vec<f64>> = vec![
      vec![9.0, 9.0],
      vec![10.0, 10.0],
      vec![11.0, 11.0],
      vec![12.0, 12.0],
    ];
    for p in &even_more {
      tree.insert(p);
    }

    let all_points = [&points[..], &more_points[..], &even_more[..]].concat();

    // Verify root CF size matches data count
    utils::verify_size(&all_points, tree.nodes().get(tree.root()).unwrap().as_cf());

    // Explicitly check root's CF against sum of children's CFs
    let root_node = &tree.nodes()[tree.root()];
    let root_cf = root_node.as_cf();
    let mut reconstructed_cf = crate::cluster_feature::VII::new(tree.dim());
    for &child_id in root_node.children() {
      reconstructed_cf.add_cf(tree.nodes()[child_id].as_cf());
    }
    let centroid_diff: f64 = root_cf
      .centroid()
      .iter()
      .zip(reconstructed_cf.centroid())
      .map(|(a, b)| (a - b).abs())
      .sum();
    assert!(
      centroid_diff < 0.001,
      "Root CF centroid mismatch: diff={}",
      centroid_diff
    );
    assert!(
      (reconstructed_cf.ssd() - root_cf.ssd()).abs() < 0.001,
      "Root CF SSD mismatch: root={}, reconstructed={}",
      root_cf.ssd(),
      reconstructed_cf.ssd()
    );

    // Verify all nodes' CFs match their children's CFs
    utils::verify_node(&tree.nodes()[tree.root()], &tree);

    // Total size check
    let total_size: usize = tree.leaf_entries().iter().map(|cf| cf.size()).sum();
    assert_eq!(
      total_size,
      all_points.len(),
      "Total leaf entry sizes ({}) should match data count ({})",
      total_size,
      all_points.len()
    );
  }

  /// Regression test: when a split propagates upward through the while-loop
  /// (grandparent has capacity), the grandparent's CF must equal the sum of
  /// its children's CFs.  Exercises the `handle_overflow_point` while-loop
  /// `Some(id)` branch where the parent is updated with `add(point)`.
  #[test]
  fn test_overflow_propagation_parent_cf_consistency() {
    use crate::cf_tree::CFNode;
    use crate::cluster_feature::ClusterFeature;

    // capacity=3, threshold=0.0 → every point gets its own leaf entry.
    // We need enough points to build a 3-level tree (root → intermediate → leaf)
    // so that a leaf overflow propagates through a full intermediate to a
    // root that still has capacity — hitting the while-loop `Some(id)` branch.
    let mut rng = StdRng::seed_from_u64(7777);
    let data: Vec<Vec<f64>> = (0..30)
      .map(|_| vec![rng.random_range(0.0..100.0), rng.random_range(0.0..100.0)])
      .collect();

    let mut tree = CFTree::<
      f64,
      crate::cluster_feature::VII<f64>,
      CentroidEuclideanDistance<f64, crate::cluster_feature::VII<f64>>,
      CentroidEuclideanDistance<f64, crate::cluster_feature::VII<f64>>,
    >::new(
      CentroidEuclideanDistance::new(),
      CentroidEuclideanDistance::new(),
      3,   // capacity
      2,   // dim
      100, // maxleaves
      0.0, // threshold — no merging, every point is its own entry
    );

    for v in data.iter() {
      tree.insert(v);
    }

    // Verify the tree has at least 3 levels (root is non-leaf with non-leaf children)
    let root = &tree.nodes()[tree.root()];
    assert!(
      !root.is_leaf(),
      "root should be a non-leaf after enough splits"
    );
    let has_non_leaf_child = root
      .children()
      .iter()
      .any(|&cid| !tree.nodes()[cid].is_leaf());
    assert!(
      has_non_leaf_child,
      "tree should have at least 3 levels (root → intermediate → leaf)"
    );

    // Every non-leaf node's CF must equal the sum of its children's CFs.
    // This is the invariant that would break if the while-loop adds `point`
    // instead of `new_node.as_cf()`.
    type TreeType = CFTree<
      f64,
      crate::cluster_feature::VII<f64>,
      CentroidEuclideanDistance<f64, crate::cluster_feature::VII<f64>>,
      CentroidEuclideanDistance<f64, crate::cluster_feature::VII<f64>>,
    >;
    fn verify_node_cf(node: &CFNode<f64, crate::cluster_feature::VII<f64>>, tree: &TreeType) {
      type VII = crate::cluster_feature::VII<f64>;
      if node.is_leaf() {
        let mut reconstructed = VII::new(tree.dim());
        for &entry_id in node.children() {
          reconstructed.add_cf(&tree.leaf_entries()[entry_id]);
        }
        let centroid_diff: f64 = node
          .as_cf()
          .centroid()
          .iter()
          .zip(reconstructed.centroid())
          .map(|(a, b)| (a - b).abs())
          .sum();
        assert!(
          centroid_diff < 0.001,
          "Leaf node centroid mismatch: node={:?}, reconstructed={:?}",
          node.as_cf().centroid(),
          reconstructed.centroid()
        );
        assert!(
          (reconstructed.ssd() - node.as_cf().ssd()).abs() < 0.001,
          "Leaf node SSD mismatch"
        );
      } else {
        let mut reconstructed = VII::new(tree.dim());
        for &child_id in node.children() {
          reconstructed.add_cf(tree.nodes()[child_id].as_cf());
        }
        let centroid_diff: f64 = node
          .as_cf()
          .centroid()
          .iter()
          .zip(reconstructed.centroid())
          .map(|(a, b)| (a - b).abs())
          .sum();
        assert!(
          centroid_diff < 0.001,
          "Non-leaf node centroid mismatch: node={:?}, reconstructed={:?}",
          node.as_cf().centroid(),
          reconstructed.centroid()
        );
        assert!(
          (reconstructed.ssd() - node.as_cf().ssd()).abs() < 0.001,
          "Non-leaf node SSD mismatch: node={}, reconstructed={}",
          node.as_cf().ssd(),
          reconstructed.ssd()
        );
        // Recurse into children
        for &child_id in node.children() {
          verify_node_cf(&tree.nodes()[child_id], tree);
        }
      }
    }

    verify_node_cf(root, &tree);

    // Also verify total point count
    let total_size: usize = tree.leaf_entries().iter().map(|cf| cf.size()).sum();
    assert_eq!(
      total_size,
      data.len(),
      "total leaf entry sizes must match data count"
    );
  }

  /// Deterministic test that forces the while-loop `Some(id)` branch with an
  /// overflow node containing multiple entries.  Uses 1D points placed so that
  /// the split assigns existing entries to the same group as the new entry,
  /// making the overflow node's CF ≠ the inserted point.
  #[test]
  fn test_overflow_propagation_multi_entry_node() {
    // 1D points, capacity=3, threshold=0.0
    // Points are placed to force a specific split pattern.
    let data: Vec<Vec<f64>> = vec![
      vec![0.0],  // A
      vec![10.0], // B
      vec![20.0], // C
      vec![30.0], // D — triggers first leaf overflow, root becomes non-leaf
      vec![1.0],  // E
      vec![2.0],  // F
      vec![3.0],  // G
      vec![4.0],  // H
      vec![5.0],  // I
      vec![6.0],  // J
      vec![7.0],  // K
      vec![8.0],  // L
      vec![9.0],  // M
      vec![15.0], // N
      vec![25.0], // O
      vec![35.0], // P
    ];

    let mut tree = CFTree::<
      f64,
      crate::cluster_feature::VII<f64>,
      CentroidEuclideanDistance<f64, crate::cluster_feature::VII<f64>>,
      CentroidEuclideanDistance<f64, crate::cluster_feature::VII<f64>>,
    >::new(
      CentroidEuclideanDistance::new(),
      CentroidEuclideanDistance::new(),
      3,   // capacity
      1,   // dim
      100, // maxleaves
      0.0, // threshold
    );

    // Insert one at a time, verifying the invariant after each insertion.
    // If the while-loop `add(point)` bug exists, it will manifest when
    // a split propagates to a grandparent that has capacity.
    for (point_idx, v) in data.iter().enumerate() {
      tree.insert(v);

      // Verify every node's CF = sum of children's CFs
      fn verify_all(
        point_idx: usize,
        node_id: usize,
        tree: &CFTree<
          f64,
          crate::cluster_feature::VII<f64>,
          CentroidEuclideanDistance<f64, crate::cluster_feature::VII<f64>>,
          CentroidEuclideanDistance<f64, crate::cluster_feature::VII<f64>>,
        >,
      ) {
        let node = &tree.nodes()[node_id];
        type VII = crate::cluster_feature::VII<f64>;
        let mut reconstructed = VII::new(tree.dim());
        if node.is_leaf() {
          for &entry_id in node.children() {
            reconstructed.add_cf(&tree.leaf_entries()[entry_id]);
          }
        } else {
          for &child_id in node.children() {
            reconstructed.add_cf(tree.nodes()[child_id].as_cf());
            verify_all(point_idx, child_id, tree);
          }
        }
        let centroid_diff: f64 = node
          .as_cf()
          .centroid()
          .iter()
          .zip(reconstructed.centroid())
          .map(|(a, b)| (a - b).abs())
          .sum();
        assert!(
          centroid_diff < 0.01,
          "Point {}: node centroid mismatch: node={:?}, reconstructed={:?}",
          point_idx,
          node.as_cf().centroid(),
          reconstructed.centroid()
        );
        assert!(
          (reconstructed.ssd() - node.as_cf().ssd()).abs() < 0.01,
          "Point {}: node SSD mismatch: node={}, reconstructed={}",
          point_idx,
          node.as_cf().ssd(),
          reconstructed.ssd()
        );
      }
      verify_all(point_idx, tree.root(), &tree);
    }

    // Final check: total point count
    let total_size: usize = tree.leaf_entries().iter().map(|cf| cf.size()).sum();
    assert_eq!(
      total_size,
      data.len(),
      "total leaf entry sizes must match data count"
    );
  }

  #[test]
  fn test_split_cfs_balanced_tie_breaking() {
    // Regression test: split_cfs must balance ties by entry count,
    // not always assign to group 1.
    use crate::cluster_feature::ClusterFeature;

    // 8 points on the perpendicular bisector of (0,0) and (10,0)
    let data: Vec<Vec<f64>> = (0..8).map(|i| vec![5.0, (i as f64) * 0.5]).collect();

    let mut tree = CFTree::new(
      CentroidEuclideanDistance::new(),
      CentroidEuclideanDistance::new(),
      4,   // capacity
      2,   // dim
      100, // maxleaves
      0.0, // threshold
    );

    for v in data.iter() {
      tree.insert(v);
    }

    // Verify CF consistency
    utils::verify_size(&data, tree.nodes().get(tree.root()).unwrap().as_cf());
    utils::verify_node(&tree.nodes()[tree.root()], &tree);

    // After inserting 8 points with capacity=4, splits should have occurred.
    let leaf_sizes: Vec<usize> = tree.leaf_entries().iter().map(|cf| cf.size()).collect();
    for (i, size) in leaf_sizes.iter().enumerate() {
      assert!(
        *size > 0,
        "Leaf node {i} has {size} entries — split was unbalanced (all equidistant points went to one group)"
      );
    }

    // Also verify: total leaf entry sizes must match data count
    let total_size: usize = leaf_sizes.iter().sum();
    assert_eq!(
      total_size,
      data.len(),
      "Total leaf entry sizes ({total_size}) should match data count ({})",
      data.len()
    );
  }
}
