//! Regression tests that exercise distance measures inside a CFTree.

#[cfg(test)]
mod tests {
  use crate::cf_tree::CFTree;
  use crate::cluster_feature::{ClusterFeature, VII};

  use super::super::{RadiusDistance, VarianceIncreaseDistance};

  #[test]
  fn test_leaf_cluster_means_variance_variance_increase() {
    // Build a tree with VarianceIncreaseDistance and check leaf CF statistics
    let data: Vec<Vec<f64>> = vec![
      vec![1.0, 2.0, 3.0],
      vec![1.1, 2.1, 3.1],
      vec![10.0, 20.0, 30.0],
      vec![10.1, 20.1, 30.1],
      vec![100.0, 200.0, 300.0],
      vec![100.1, 200.1, 300.1],
    ];
    let mut tree = CFTree::<
      f64,
      VII<f64>,
      VarianceIncreaseDistance<f64, VII<f64>>,
      VarianceIncreaseDistance<f64, VII<f64>>,
    >::new(
      VarianceIncreaseDistance::new(),
      VarianceIncreaseDistance::new(),
      10,
      3,
      100,
      f64::MAX,
    );
    for row in &data {
      tree.insert(row);
    }

    // Verify total points preserved
    let total: usize = tree.leaf_entries().iter().map(|cf| cf.size()).sum();
    assert_eq!(total, data.len());

    // Each leaf should have a well-defined mean and variance
    for cf in tree.leaf_entries() {
      assert!(cf.size() > 0, "leaf should have points");
      assert!(cf.ssd() >= 0.0, "ssd should be non-negative");
      let variance = cf.ssd() / (cf.size() as f64);
      assert!(variance >= 0.0, "variance should be non-negative");
    }
  }

  #[test]
  fn test_leaf_cluster_means_variance_radius() {
    let data: Vec<Vec<f64>> = vec![
      vec![1.0, 2.0, 3.0],
      vec![1.1, 2.1, 3.1],
      vec![10.0, 20.0, 30.0],
      vec![10.1, 20.1, 30.1],
      vec![100.0, 200.0, 300.0],
      vec![100.1, 200.1, 300.1],
    ];
    let mut tree =
      CFTree::<f64, VII<f64>, RadiusDistance<f64, VII<f64>>, RadiusDistance<f64, VII<f64>>>::new(
        RadiusDistance::new(),
        RadiusDistance::new(),
        10,
        3,
        100,
        f64::MAX,
      );
    for row in &data {
      tree.insert(row);
    }

    let total: usize = tree.leaf_entries().iter().map(|cf| cf.size()).sum();
    assert_eq!(total, data.len());

    for cf in tree.leaf_entries() {
      assert!(cf.size() > 0, "leaf should have points");
      assert!(cf.ssd() >= 0.0, "ssd should be non-negative");
      let variance = cf.ssd() / (cf.size() as f64);
      assert!(variance >= 0.0, "variance should be non-negative");
    }
  }

  #[test]
  fn test_leaf_cluster_regression_variance_increase() {
    // Deterministic data with known cluster structure
    // Tight clusters near origin and near (10,10)
    let data: Vec<Vec<f64>> = vec![
      vec![0.0, 0.0],
      vec![0.1, 0.1],
      vec![0.2, 0.2],
      vec![10.0, 10.0],
      vec![10.1, 10.1],
      vec![10.2, 10.2],
    ];
    // Use small maxleaves and small threshold to force separate clusters
    let mut tree = CFTree::<
      f64,
      VII<f64>,
      VarianceIncreaseDistance<f64, VII<f64>>,
      VarianceIncreaseDistance<f64, VII<f64>>,
    >::new(
      VarianceIncreaseDistance::new(),
      VarianceIncreaseDistance::new(),
      2,
      2,
      3,
      1.0,
    );
    for row in &data {
      tree.insert(row);
    }

    // Collect leaf statistics
    let mut leaves: Vec<(usize, f64, f64)> = tree
      .leaf_entries()
      .iter()
      .map(|cf| {
        let mean_val = cf.centroid().iter().sum::<f64>() / cf.centroid().len() as f64;
        let variance = cf.ssd() / cf.size() as f64;
        (cf.size(), mean_val, variance)
      })
      .collect();
    leaves.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    // Total points must match
    let total: usize = leaves.iter().map(|l| l.0).sum();
    assert_eq!(total, data.len());

    // All variances should be small (tight clusters)
    for (_size, _mean, var) in &leaves {
      assert!(*var >= 0.0 && *var < 1.0, "variance {} out of range", var);
    }
  }

  #[test]
  fn test_leaf_cluster_regression_radius() {
    let data: Vec<Vec<f64>> = vec![
      vec![0.0, 0.0],
      vec![0.1, 0.1],
      vec![0.2, 0.2],
      vec![10.0, 10.0],
      vec![10.1, 10.1],
      vec![10.2, 10.2],
    ];
    let mut tree = CFTree::<
      f64,
      VII<f64>,
      RadiusDistance<f64, VII<f64>>,
      RadiusDistance<f64, VII<f64>>,
    >::new(RadiusDistance::new(), RadiusDistance::new(), 2, 2, 3, 1.0);
    for row in &data {
      tree.insert(row);
    }

    let mut leaves: Vec<(usize, f64, f64)> = tree
      .leaf_entries()
      .iter()
      .map(|cf| {
        let mean_val = cf.centroid().iter().sum::<f64>() / cf.centroid().len() as f64;
        let variance = cf.ssd() / cf.size() as f64;
        (cf.size(), mean_val, variance)
      })
      .collect();
    leaves.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    let total: usize = leaves.iter().map(|l| l.0).sum();
    assert_eq!(total, data.len());

    for (_size, _mean, var) in &leaves {
      assert!(*var >= 0.0 && *var < 1.0, "variance {} out of range", var);
    }
  }
}
