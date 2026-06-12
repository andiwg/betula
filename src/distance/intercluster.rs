use std::marker::PhantomData;

use crate::{cluster_feature::ClusterFeature, types::Float};

use super::{CFDistance, SqEuclidean, VectorDistance};

// ── Average Intercluster Distance (Zhang "D2") ──

/// Average intercluster distance (D2).
///
/// Measures the average squared Euclidean distance between a point and all
/// points in a cluster. Unlike centroid distance (which ignores cluster
/// spread), this accounts for cluster variance, making it better at avoiding
/// merging spread-out clusters.
///
/// Formulas (derived from ELKI):
///   sq_dist(CF, v)  = ||v - centroid||² + ssd/n
///   sq_dist(CF1, CF2) = ||c1 - c2||² + ssd1/n1 + ssd2/n2
///
/// References:
///
/// Andreas Lang and Erich Schubert
/// "BETULA: Numerically Stable CF-Trees for BIRCH Clustering"
/// Int. Conf. on Similarity Search and Applications (SISAP) 2020
///
/// Andreas Lang and Erich Schubert
/// "BETULA: Fast Clustering of Large Data with Improved BIRCH CF-Trees"
/// Information Systems 2022
///
/// T. Zhang, "Data Clustering for Very Large Datasets"
/// University of Wisconsin Madison, Technical Report #1355, 1997
pub struct AverageInterclusterDistance<F: Float, C: ClusterFeature<F>> {
  _marker: PhantomData<C>,
  _marker2: PhantomData<F>,
}
impl<F: Float, C: ClusterFeature<F>> Clone for AverageInterclusterDistance<F, C> {
  fn clone(&self) -> Self {
    Self::new()
  }
}
impl<F: Float, C: ClusterFeature<F>> Default for AverageInterclusterDistance<F, C> {
  fn default() -> Self {
    Self::new()
  }
}

impl<F: Float, C: ClusterFeature<F>> AverageInterclusterDistance<F, C> {
  pub fn new() -> Self {
    AverageInterclusterDistance {
      _marker: PhantomData,
      _marker2: PhantomData,
    }
  }
}
impl<F: Float, C: ClusterFeature<F>> CFDistance<F, C> for AverageInterclusterDistance<F, C> {
  fn sq_dist(&self, cf1: &C, x: &[F], d: usize) -> F {
    let n = cf1.size();
    if n == 0 {
      return F::zero();
    }
    // ||v - centroid||² + ssd/n
    let centroid_dist = SqEuclidean::dist(cf1.centroid(), x, d);
    let variance = cf1.ssd() / F::from_index(n);
    centroid_dist + variance
  }

  fn sq_dist_cf(&self, cf1: &C, cf2: &C, d: usize) -> F {
    let n1 = cf1.size();
    let n2 = cf2.size();
    if n1 == 0 || n2 == 0 {
      return F::zero();
    }
    // ||c1 - c2||² + ssd1/n1 + ssd2/n2
    let centroid_dist = SqEuclidean::dist(cf1.centroid(), cf2.centroid(), d);
    let var1 = cf1.ssd() / F::from_index(n1);
    let var2 = cf2.ssd() / F::from_index(n2);
    centroid_dist + var1 + var2
  }
}

// ── Average Intracluster Distance (Lang & Schubert "D3") ──

/// Average intracluster distance (D3).
///
/// Measures the average squared Euclidean distance between all pairs of points
/// within a cluster or between two clusters.
///
/// Formulas (from ELKI AverageIntraclusterDistance):
///   sq_dist(CF, v)  = 2 * ((n+1)*ssd + n*||c - v||²) / (n*(n+1))
///   sq_dist(CF1, CF2) = 2 * (n12*(ssd1+ssd2) + n1*n2*||c1-c2||²) / (n12*(n12-1))
///
/// References:
///
/// Andreas Lang and Erich Schubert
/// "BETULA: Numerically Stable CF-Trees for BIRCH Clustering"
/// Int. Conf. on Similarity Search and Applications (SISAP) 2020
///
/// Andreas Lang and Erich Schubert
/// "BETULA: Fast Clustering of Large Data with Improved BIRCH CF-Trees"
/// Information Systems 2022
pub struct AverageIntraclusterDistance<F: Float, C: ClusterFeature<F>> {
  _marker: PhantomData<C>,
  _marker2: PhantomData<F>,
}
impl<F: Float, C: ClusterFeature<F>> Clone for AverageIntraclusterDistance<F, C> {
  fn clone(&self) -> Self {
    Self::new()
  }
}
impl<F: Float, C: ClusterFeature<F>> Default for AverageIntraclusterDistance<F, C> {
  fn default() -> Self {
    Self::new()
  }
}

impl<F: Float, C: ClusterFeature<F>> AverageIntraclusterDistance<F, C> {
  pub fn new() -> Self {
    AverageIntraclusterDistance {
      _marker: PhantomData,
      _marker2: PhantomData,
    }
  }
}
impl<F: Float, C: ClusterFeature<F>> CFDistance<F, C> for AverageIntraclusterDistance<F, C> {
  fn sq_dist(&self, cf1: &C, x: &[F], d: usize) -> F {
    let n = cf1.size();
    if n == 0 {
      return F::zero();
    }
    // 2 * ((n+1)*ssd + n*||c - x||²) / (n*(n+1))
    let n_f = F::from_index(n);
    let n_plus_1 = n_f + F::one();
    let centroid_dist = SqEuclidean::dist(cf1.centroid(), x, d);
    let ssd = cf1.ssd();
    // Use FMA for numerical stability: 2 * ((n+1)*ssd + n*centroid_dist²) / (n*(n+1))
    let two = F::one() + F::one();
    let numerator = n_plus_1.mul_add(ssd, n_f * centroid_dist);
    let denominator = n_f * n_plus_1;
    two * numerator / denominator
  }

  fn sq_dist_cf(&self, cf1: &C, cf2: &C, d: usize) -> F {
    let n1 = cf1.size();
    let n2 = cf2.size();
    if n1 == 0 || n2 == 0 {
      return F::zero();
    }
    // 2 * (n12*(ssd1+ssd2) + n1*n2*||c1-c2||²) / (n12*(n12-1))
    let n1_f = F::from_index(n1);
    let n2_f = F::from_index(n2);
    let n12_f = n1_f + n2_f;
    let centroid_dist = SqEuclidean::dist(cf1.centroid(), cf2.centroid(), d);
    let ssd_sum = cf1.ssd() + cf2.ssd();
    let n1n2 = n1_f * n2_f;
    // Use FMA for numerical stability
    let two = F::one() + F::one();
    let numerator = n12_f.mul_add(ssd_sum, n1n2 * centroid_dist);
    let denominator = n12_f * (n12_f - F::one());
    two * numerator / denominator
  }
}

// ── Tests ──

#[cfg(test)]
mod tests {
  use crate::cluster_feature::VII;

  use super::{super::CentroidEuclideanDistance, *};

  macro_rules! assert_close {
        ($a:expr, $b:expr) => {
            assert!(
                ($a - $b).abs() < 1e-10,
                "expected={}, actual={}",
                $a,
                $b
            );
        };
        ($a:expr, $b:expr, $($arg:tt)*) => {
            assert!(
                ($a - $b).abs() < 1e-10,
                $($arg)*
            );
        };
    }

  // ── AverageInterclusterDistance tests ──

  /// Reference: avg intercluster dist = (1/n) * Σ||v - p_i||² = ||v - centroid||² + ssd/n
  fn reference_avg_intercluster(points: &[&[f64]], v: &[f64], d: usize) -> f64 {
    let n = points.len();
    if n == 0 {
      return 0.0;
    }
    let mut sum = 0.0;
    for p in points {
      for i in 0..d {
        let diff = v[i] - p[i];
        sum += diff * diff;
      }
    }
    sum / n as f64
  }

  /// Reference: avg intercluster dist between two CFs
  /// = (1/(n1*n2)) * Σ_i Σ_j ||p1_i - p2_j||² = ||c1 - c2||² + ssd1/n1 + ssd2/n2
  fn reference_avg_intercluster_cf(points1: &[&[f64]], points2: &[&[f64]], d: usize) -> f64 {
    let n1 = points1.len();
    let n2 = points2.len();
    if n1 == 0 || n2 == 0 {
      return 0.0;
    }
    let mut sum = 0.0;
    for p1 in points1 {
      for p2 in points2 {
        for i in 0..d {
          let diff = p1[i] - p2[i];
          sum += diff * diff;
        }
      }
    }
    sum / (n1 * n2) as f64
  }

  #[test]
  fn test_avg_intercluster() {
    let aid = AverageInterclusterDistance::<f64, VII<f64>>::new();

    // CF with points: [1,2,3], [3,4,5], [5,6,7] -> centroid=[3,4,5], ssd=24
    let mut cf = VII::<f64>::new(3);
    cf.add(&[1.0, 2.0, 3.0]);
    cf.add(&[3.0, 4.0, 5.0]);
    cf.add(&[5.0, 6.0, 7.0]);

    let points: [f64; 9] = [1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 5.0, 6.0, 7.0];
    let rows: &[&[f64]] = &[&points[0..3], &points[3..6], &points[6..9]];

    // Query [0,0,0]: 50 + 8 = 58
    assert_close!(
      reference_avg_intercluster(rows, &[0.0, 0.0, 0.0], 3),
      aid.sq_dist(&cf, &[0.0, 0.0, 0.0], 3)
    );
    // Query centroid [3,4,5]: 0 + 8 = 8
    assert_close!(
      reference_avg_intercluster(rows, &[3.0, 4.0, 5.0], 3),
      aid.sq_dist(&cf, &[3.0, 4.0, 5.0], 3)
    );
  }

  #[test]
  fn test_avg_intercluster_cf_to_cf() {
    let aid = AverageInterclusterDistance::<f64, VII<f64>>::new();

    let mut cf1 = VII::<f64>::new(2);
    cf1.add(&[1.0, 2.0]);
    cf1.add(&[3.0, 4.0]);

    let mut cf2 = VII::<f64>::new(2);
    cf2.add(&[5.0, 6.0]);
    cf2.add(&[7.0, 8.0]);

    // ||[2,3]-[6,7]||² + 2 + 2 = 32+2+2 = 36
    let p1: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
    let p2: [f64; 4] = [5.0, 6.0, 7.0, 8.0];
    let rows1: &[&[f64]] = &[&p1[0..2], &p1[2..4]];
    let rows2: &[&[f64]] = &[&p2[0..2], &p2[2..4]];
    let expected = reference_avg_intercluster_cf(rows1, rows2, 2);
    assert_close!(expected, aid.sq_dist_cf(&cf1, &cf2, 2));
  }

  #[test]
  fn test_avg_intercluster_edge_cases() {
    let aid = AverageInterclusterDistance::<f64, VII<f64>>::new();

    // Empty CF returns 0
    let empty = VII::<f64>::new(2);
    assert_eq!(aid.sq_dist(&empty, &[1.0, 2.0], 2), 0.0);
    assert_eq!(aid.sq_dist_cf(&empty, &empty, 2), 0.0);

    // Single point CF: ssd=0, so avg_intercluster == centroid_euclidean
    let mut single = VII::<f64>::new(2);
    single.add(&[3.0, 4.0]);
    assert_close!(25.0, aid.sq_dist(&single, &[0.0, 0.0], 2));

    // Identical points -> ssd=0 -> avg_intercluster == centroid_euclidean
    let mut identical = VII::<f64>::new(2);
    identical.add(&[3.0, 4.0]);
    identical.add(&[3.0, 4.0]);
    identical.add(&[3.0, 4.0]);
    let ce = CentroidEuclideanDistance::<f64, VII<f64>>::new();
    assert_close!(
      ce.sq_dist(&identical, &[0.0, 0.0], 2),
      aid.sq_dist(&identical, &[0.0, 0.0], 2)
    );
  }

  #[test]
  fn test_avg_intercluster_various_dimensions() {
    let dims = [1, 2, 5, 8, 16];
    let points: Vec<Vec<f64>> = vec![
      (0..16).map(|i| (i + 1) as f64).collect(),
      (0..16).map(|i| (i + 2) as f64).collect(),
      (0..16).map(|i| i as f64).collect(),
    ];
    let query: Vec<f64> = (0..16).map(|i| (i as f64) * 0.5).collect();

    for &dim in &dims {
      let mut cf = VII::<f64>::new(dim);
      for p in &points {
        cf.add(&p[..dim]);
      }

      let aid = AverageInterclusterDistance::<f64, VII<f64>>::new();
      let expected = reference_avg_intercluster(
        &points.iter().map(|p| &p[..dim]).collect::<Vec<_>>(),
        &query[..dim],
        dim,
      );
      assert_close!(
        expected,
        aid.sq_dist(&cf, &query[..dim], dim),
        "dim={}",
        dim
      );
    }
  }

  // ── AverageIntraclusterDistance tests ──

  /// Reference using the ELKI formula:
  /// sq_dist(CF, v) = 2 * ((n+1)*ssd + n*||c - v||²) / (n*(n+1))
  fn reference_avg_intracluster(cf_ssd: f64, centroid: &[f64], v: &[f64], d: usize) -> f64 {
    let mut centroid_dist_sq = 0.0;
    for i in 0..d {
      let diff = centroid[i] - v[i];
      centroid_dist_sq += diff * diff;
    }
    let n_points = 3.0;
    let n_f = n_points;
    let n_plus_1 = n_f + 1.0;
    let two = 2.0;
    (two * (n_plus_1 * cf_ssd + n_f * centroid_dist_sq)) / (n_f * n_plus_1)
  }

  /// Reference for CF-to-CF using ELKI formula:
  /// sq_dist_cf(CF1, CF2) = 2 * (n12*(ssd1+ssd2) + n1*n2*||c1-c2||²) / (n12*(n12-1))
  fn reference_avg_intracluster_cf(
    ssd1: f64,
    centroid1: &[f64],
    n1: usize,
    ssd2: f64,
    centroid2: &[f64],
    n2: usize,
    d: usize,
  ) -> f64 {
    let n1f = n1 as f64;
    let n2f = n2 as f64;
    let n12f = n1f + n2f;

    let mut centroid_dist_sq = 0.0;
    for i in 0..d {
      let diff = centroid1[i] - centroid2[i];
      centroid_dist_sq += diff * diff;
    }

    let two = 2.0;
    let numerator = n12f * (ssd1 + ssd2) + n1f * n2f * centroid_dist_sq;
    let denominator = n12f * (n12f - 1.0);
    two * numerator / denominator
  }

  #[test]
  fn test_avg_intracluster() {
    let aid = AverageIntraclusterDistance::<f64, VII<f64>>::new();

    // CF with points: [1,2,3], [3,4,5], [5,6,7] -> centroid=[3,4,5], ssd=24
    let mut cf = VII::<f64>::new(3);
    cf.add(&[1.0, 2.0, 3.0]);
    cf.add(&[3.0, 4.0, 5.0]);
    cf.add(&[5.0, 6.0, 7.0]);

    // Verify CF statistics
    assert_eq!(3, cf.size());
    assert_close!(24.0, cf.ssd());

    // Query [0,0,0]: 2*(4*24 + 3*50)/(3*4) = 2*246/12 = 41
    let expected = reference_avg_intracluster(24.0, &[3.0, 4.0, 5.0], &[0.0, 0.0, 0.0], 3);
    assert_close!(
      expected,
      aid.sq_dist(&cf, &[0.0, 0.0, 0.0], 3),
      "query=[0,0,0], expected={}",
      expected
    );

    // Query centroid [3,4,5]: 2*(4*24 + 3*0)/(3*4) = 2*96/12 = 16
    let expected2 = reference_avg_intracluster(24.0, &[3.0, 4.0, 5.0], &[3.0, 4.0, 5.0], 3);
    assert_close!(
      expected2,
      aid.sq_dist(&cf, &[3.0, 4.0, 5.0], 3),
      "query=centroid, expected={}",
      expected2
    );
  }

  #[test]
  fn test_avg_intracluster_cf_to_cf() {
    let aid = AverageIntraclusterDistance::<f64, VII<f64>>::new();

    // CF1 with points: [1,2], [3,4] -> centroid=[2,3], ssd = 4+4 = 8
    let mut cf1 = VII::<f64>::new(2);
    cf1.add(&[1.0, 2.0]);
    cf1.add(&[3.0, 4.0]);

    // CF2 with points: [5,6], [7,8] -> centroid=[6,7], ssd = 4+4 = 8
    let mut cf2 = VII::<f64>::new(2);
    cf2.add(&[5.0, 6.0]);
    cf2.add(&[7.0, 8.0]);

    // ||c1 - c2||² = ||[2,3] - [6,7]||² = 16+16 = 32
    // n12 = 4, n1 = n2 = 2
    // Formula: 2 * (4*(4+4) + 2*2*32) / (4*3) = 2*(32+128) / 12 = 320/12 = 26.666...
    let expected = reference_avg_intracluster_cf(4.0, &[2.0, 3.0], 2, 4.0, &[6.0, 7.0], 2, 2);
    assert_close!(expected, aid.sq_dist_cf(&cf1, &cf2, 2));
  }

  #[test]
  fn test_avg_intracluster_edge_cases() {
    let aid = AverageIntraclusterDistance::<f64, VII<f64>>::new();

    // Empty CF returns 0
    let empty = VII::<f64>::new(2);
    assert_eq!(aid.sq_dist(&empty, &[1.0, 2.0], 2), 0.0);
    assert_eq!(aid.sq_dist_cf(&empty, &empty, 2), 0.0);

    // Single point CF: distance to the same point = 0
    let mut single = VII::<f64>::new(2);
    single.add(&[3.0, 4.0]);
    assert_close!(0.0, aid.sq_dist(&single, &[3.0, 4.0], 2));

    // Distance to origin: ||c||² = 25, ssd = 0, n = 1
    // Formula: 2 * (2*0 + 1*25) / 2 = 2*25/2 = 25
    assert_close!(25.0, aid.sq_dist(&single, &[0.0, 0.0], 2));
  }

  #[test]
  fn test_avg_intracluster_single_point_equivalence() {
    // For single-point clusters, AverageIntraclusterDistance equals CentroidEuclideanDistance
    let mut cf1 = VII::<f64>::new(2);
    cf1.add(&[1.0, 2.0]);
    let mut cf2 = VII::<f64>::new(2);
    cf2.add(&[4.0, 6.0]);

    let intra = AverageIntraclusterDistance::<f64, VII<f64>>::new();
    let ce = CentroidEuclideanDistance::<f64, VII<f64>>::new();

    // For single-point CFs, centroid distance = point distance
    // Both should give ||[1,2] - [4,6]||² = 9+16 = 25
    assert_close!(
      ce.sq_dist_cf(&cf1, &cf2, 2),
      intra.sq_dist_cf(&cf1, &cf2, 2)
    );

    // For query at the single point, distance is 0
    assert_close!(0.0, intra.sq_dist(&cf1, &[1.0, 2.0], 2));
  }
}
