use std::marker::PhantomData;

use crate::{cluster_feature::ClusterFeature, types::Float};

use super::{CFDistance, Manhattan, SqEuclidean, VectorDistance};

// ── Centroid Euclidean Distance (D0) ──

/// Centroid Euclidean distance.
///
/// Measures the squared Euclidean distance between a point (or centroid) and
/// the centroid of a cluster feature. This is the simplest BIRCH distance
/// criterion and ignores cluster spread.
///
/// Formulas:
///   sq_dist(CF, v)   = ||v - centroid||²
///   sq_dist(CF1, CF2) = ||centroid1 - centroid2||²
///
/// Reference:
///
/// Andreas Lang and Erich Schubert
/// "BETULA: Numerically Stable CF-Trees for BIRCH Clustering"
/// Int. Conf. on Similarity Search and Applications (SISAP) 2020
///
/// Andreas Lang and Erich Schubert
/// "BETULA: Fast Clustering of Large Data with Improved BIRCH CF-Trees"
/// Information Systems 2022
pub struct CentroidEuclideanDistance<F: Float, C: ClusterFeature<F>> {
  _marker: PhantomData<C>,
  _marker2: PhantomData<F>,
}
impl<F: Float, C: ClusterFeature<F>> Clone for CentroidEuclideanDistance<F, C> {
  fn clone(&self) -> Self {
    Self::new()
  }
}
impl<F: Float, C: ClusterFeature<F>> Default for CentroidEuclideanDistance<F, C> {
  fn default() -> Self {
    Self::new()
  }
}

impl<F: Float, C: ClusterFeature<F>> CentroidEuclideanDistance<F, C> {
  pub fn new() -> Self {
    CentroidEuclideanDistance {
      _marker: PhantomData,
      _marker2: PhantomData,
    }
  }
}
impl<F: Float, C: ClusterFeature<F>> CFDistance<F, C> for CentroidEuclideanDistance<F, C> {
  fn sq_dist(&self, cf1: &C, x: &[F], d: usize) -> F {
    SqEuclidean::dist(cf1.centroid(), x, d)
  }

  fn sq_dist_cf(&self, cf1: &C, cf2: &C, d: usize) -> F {
    SqEuclidean::dist(cf1.centroid(), cf2.centroid(), d)
  }
}

// ── Centroid Manhattan Distance (D1) ──

/// Centroid Manhattan distance.
///
/// Measures the Manhattan (L1) distance between a point (or centroid) and
/// the centroid of a cluster feature. Like centroid Euclidean, this ignores
/// cluster spread.
///
/// Formulas:
///   sq_dist(CF, v)   = Σᵢ |vᵢ - centroidᵢ|
///   sq_dist(CF1, CF2) = Σᵢ |centroid1ᵢ - centroid2ᵢ|
///
/// Reference:
///
/// Andreas Lang and Erich Schubert
/// "BETULA: Numerically Stable CF-Trees for BIRCH Clustering"
/// Int. Conf. on Similarity Search and Applications (SISAP) 2020
///
/// Andreas Lang and Erich Schubert
/// "BETULA: Fast Clustering of Large Data with Improved BIRCH CF-Trees"
/// Information Systems 2022
pub struct CentroidManhattanDistance<F: Float, C: ClusterFeature<F>> {
  _marker: PhantomData<C>,
  _marker2: PhantomData<F>,
}
impl<F: Float, C: ClusterFeature<F>> Clone for CentroidManhattanDistance<F, C> {
  fn clone(&self) -> Self {
    Self::new()
  }
}
impl<F: Float, C: ClusterFeature<F>> Default for CentroidManhattanDistance<F, C> {
  fn default() -> Self {
    Self::new()
  }
}

impl<F: Float, C: ClusterFeature<F>> CentroidManhattanDistance<F, C> {
  pub fn new() -> Self {
    CentroidManhattanDistance {
      _marker: PhantomData,
      _marker2: PhantomData,
    }
  }
}
impl<F: Float, C: ClusterFeature<F>> CFDistance<F, C> for CentroidManhattanDistance<F, C> {
  fn sq_dist(&self, cf1: &C, x: &[F], d: usize) -> F {
    Manhattan::dist(cf1.centroid(), x, d)
  }

  fn sq_dist_cf(&self, cf1: &C, cf2: &C, d: usize) -> F {
    Manhattan::dist(cf1.centroid(), cf2.centroid(), d)
  }
}

// ── Tests ──

#[cfg(test)]
mod tests {
  use crate::cluster_feature::VII;

  use super::*;

  macro_rules! assert_close {
    ($a:expr, $b:expr) => {
      assert!(($a - $b).abs() < 1e-10, "expected={}, actual={}", $a, $b);
    };
  }

  #[test]
  fn test_centroid_distances() {
    let mut cf1 = VII::<f64>::new(3);
    cf1.add(&[1.0, 2.0, 3.0]);
    let mut cf2 = VII::<f64>::new(3);
    cf2.add(&[4.0, 0.0, 1.0]);

    // CentroidEuclidean: ||[1,2,3] - [4,0,1]||² = 9+4+4 = 17
    let ce = CentroidEuclideanDistance::<f64, VII<f64>>::new();
    assert_close!(17.0, ce.sq_dist_cf(&cf1, &cf2, 3));
    assert_close!(14.0, ce.sq_dist(&cf1, &[0.0, 0.0, 0.0], 3));

    // CentroidManhattan: |1-4| + |2-0| + |3-1| = 3+2+2 = 7
    let cm = CentroidManhattanDistance::<f64, VII<f64>>::new();
    assert_close!(7.0, cm.sq_dist_cf(&cf1, &cf2, 3));
    assert_close!(6.0, cm.sq_dist(&cf1, &[0.0, 0.0, 0.0], 3));
  }
}
