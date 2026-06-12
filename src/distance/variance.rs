use std::marker::PhantomData;

use crate::{cluster_feature::ClusterFeature, types::Float};

use super::{CFDistance, SqEuclidean, VectorDistance};

/// Variance Increase distance (D4).
///
/// Measures the increase in Sum of Squared Deviations (SSD) that would result
/// from adding a point to a cluster or merging two clusters. This is the
/// classic BIRCH splitting criterion.
///
/// Formulas (derived from ELKI VarianceIncreaseDistance):
///   sq_dist(CF, v)  = ||v - centroid||² * n / (n+1)
///   sq_dist(CF1, CF2) = ||c1 - c2||² * n1 * n2 / (n1 + n2)
///
/// The SSD increase is always less than d² because the centroid shifts toward
/// the new point.
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
pub struct VarianceIncreaseDistance<F: Float, C: ClusterFeature<F>> {
  _marker: PhantomData<C>,
  _marker2: PhantomData<F>,
}
impl<F: Float, C: ClusterFeature<F>> Clone for VarianceIncreaseDistance<F, C> {
  fn clone(&self) -> Self {
    Self::new()
  }
}
impl<F: Float, C: ClusterFeature<F>> Default for VarianceIncreaseDistance<F, C> {
  fn default() -> Self {
    Self::new()
  }
}

impl<F: Float, C: ClusterFeature<F>> VarianceIncreaseDistance<F, C> {
  pub fn new() -> Self {
    VarianceIncreaseDistance {
      _marker: PhantomData,
      _marker2: PhantomData,
    }
  }
}
impl<F: Float, C: ClusterFeature<F>> CFDistance<F, C> for VarianceIncreaseDistance<F, C> {
  fn sq_dist(&self, cf1: &C, x: &[F], d: usize) -> F {
    let n = cf1.size();
    if n == 0 {
      return F::zero();
    }
    // ||v - centroid||² * n / (n+1)
    let centroid_dist = SqEuclidean::dist(cf1.centroid(), x, d);
    let n_f = F::from_index(n);
    let n_plus_1 = n_f + F::one();
    centroid_dist * n_f / n_plus_1
  }

  fn sq_dist_cf(&self, cf1: &C, cf2: &C, d: usize) -> F {
    let n1 = cf1.size();
    let n2 = cf2.size();
    if n1 == 0 || n2 == 0 {
      return F::zero();
    }
    // ||c1 - c2||² * n1 * n2 / (n1 + n2)
    let centroid_dist = SqEuclidean::dist(cf1.centroid(), cf2.centroid(), d);
    let n1_f = F::from_index(n1);
    let n2_f = F::from_index(n2);
    let n12_f = n1_f + n2_f;
    centroid_dist * n1_f * n2_f / n12_f
  }
}

// ── Tests ──

#[cfg(test)]
mod tests {
  use crate::cluster_feature::VII;

  use super::*;

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

  /// Reference: variance increase when adding v to a cluster = ||v - centroid||² * n / (n+1)
  fn reference_variance_increase(points: &[&[f64]], v: &[f64], d: usize) -> f64 {
    let n = points.len();
    if n == 0 {
      return 0.0;
    }
    // Compute centroid
    let mut centroid = vec![0.0; d];
    for p in points {
      for i in 0..d {
        centroid[i] += p[i];
      }
    }
    for i in 0..d {
      centroid[i] /= n as f64;
    }
    // ||v - centroid||² * n / (n+1)
    let mut dist_sq = 0.0;
    for i in 0..d {
      let diff = v[i] - centroid[i];
      dist_sq += diff * diff;
    }
    dist_sq * n as f64 / (n + 1) as f64
  }

  /// Reference: variance increase when merging two clusters = ||c1-c2||² * n1*n2/(n1+n2)
  fn reference_variance_increase_cf(points1: &[&[f64]], points2: &[&[f64]], d: usize) -> f64 {
    let n1 = points1.len();
    let n2 = points2.len();
    if n1 == 0 || n2 == 0 {
      return 0.0;
    }
    let mut c1 = vec![0.0; d];
    let mut c2 = vec![0.0; d];
    for p in points1 {
      for i in 0..d {
        c1[i] += p[i];
      }
    }
    for p in points2 {
      for i in 0..d {
        c2[i] += p[i];
      }
    }
    for i in 0..d {
      c1[i] /= n1 as f64;
      c2[i] /= n2 as f64;
    }
    let mut dist_sq = 0.0;
    for i in 0..d {
      let diff = c1[i] - c2[i];
      dist_sq += diff * diff;
    }
    dist_sq * n1 as f64 * n2 as f64 / (n1 + n2) as f64
  }

  /// Reference: verify by computing actual SSD increase
  fn reference_variance_increase_by_ssd(points: &[&[f64]], v: &[f64], d: usize) -> f64 {
    let n = points.len();
    if n == 0 {
      return 0.0;
    }
    // Old SSD
    let mut old_centroid = vec![0.0; d];
    for p in points {
      for i in 0..d {
        old_centroid[i] += p[i];
      }
    }
    for i in 0..d {
      old_centroid[i] /= n as f64;
    }
    let mut old_ssd = 0.0;
    for p in points {
      for i in 0..d {
        let diff = p[i] - old_centroid[i];
        old_ssd += diff * diff;
      }
    }
    // New SSD after adding v
    let mut new_centroid = old_centroid.clone();
    for i in 0..d {
      new_centroid[i] = (old_centroid[i] * n as f64 + v[i]) / (n + 1) as f64;
    }
    let mut new_ssd = 0.0;
    for p in points {
      for i in 0..d {
        let diff = p[i] - new_centroid[i];
        new_ssd += diff * diff;
      }
    }
    for i in 0..d {
      let diff = v[i] - new_centroid[i];
      new_ssd += diff * diff;
    }
    new_ssd - old_ssd
  }

  #[test]
  fn test_variance_increase() {
    let vid = VarianceIncreaseDistance::<f64, VII<f64>>::new();

    // CF with points: [1,2,3], [3,4,5], [5,6,7] -> centroid=[3,4,5], ssd=24
    let mut cf = VII::<f64>::new(3);
    cf.add(&[1.0, 2.0, 3.0]);
    cf.add(&[3.0, 4.0, 5.0]);
    cf.add(&[5.0, 6.0, 7.0]);

    let points: [f64; 9] = [1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 5.0, 6.0, 7.0];
    let rows: &[&[f64]] = &[&points[0..3], &points[3..6], &points[6..9]];

    // Query [0,0,0]: ||[0,0,0]-[3,4,5]||² * 3/4 = 50 * 0.75 = 37.5
    let expected = reference_variance_increase(rows, &[0.0, 0.0, 0.0], 3);
    assert_close!(
      expected,
      vid.sq_dist(&cf, &[0.0, 0.0, 0.0], 3),
      "query=[0,0,0], expected={}",
      expected
    );

    // Query centroid [3,4,5]: 0 * 3/4 = 0
    assert_close!(0.0, vid.sq_dist(&cf, &[3.0, 4.0, 5.0], 3));

    // Verify against SSD-based reference
    let ssd_ref = reference_variance_increase_by_ssd(rows, &[0.0, 0.0, 0.0], 3);
    assert_close!(
      ssd_ref,
      vid.sq_dist(&cf, &[0.0, 0.0, 0.0], 3),
      "SSD reference={}",
      ssd_ref
    );
  }

  #[test]
  fn test_variance_increase_cf_to_cf() {
    let vid = VarianceIncreaseDistance::<f64, VII<f64>>::new();

    let mut cf1 = VII::<f64>::new(2);
    cf1.add(&[1.0, 2.0]);
    cf1.add(&[3.0, 4.0]);

    let mut cf2 = VII::<f64>::new(2);
    cf2.add(&[5.0, 6.0]);
    cf2.add(&[7.0, 8.0]);

    // ||c1-c2||² = ||[2,3]-[6,7]||² = 32, n1=n2=2, n1*n2/(n1+n2) = 4/4 = 1
    // Result: 32 * 1 = 32
    let p1: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
    let p2: [f64; 4] = [5.0, 6.0, 7.0, 8.0];
    let rows1: &[&[f64]] = &[&p1[0..2], &p1[2..4]];
    let rows2: &[&[f64]] = &[&p2[0..2], &p2[2..4]];
    let expected = reference_variance_increase_cf(rows1, rows2, 2);
    assert_close!(expected, vid.sq_dist_cf(&cf1, &cf2, 2));
  }

  #[test]
  fn test_variance_increase_edge_cases() {
    let vid = VarianceIncreaseDistance::<f64, VII<f64>>::new();

    // Empty CF returns 0
    let empty = VII::<f64>::new(2);
    assert_eq!(vid.sq_dist(&empty, &[1.0, 2.0], 2), 0.0);
    assert_eq!(vid.sq_dist_cf(&empty, &empty, 2), 0.0);

    // Single point CF: n/(n+1) = 1/2
    // Distance to same point: 0 * 1/2 = 0
    let mut single = VII::<f64>::new(2);
    single.add(&[3.0, 4.0]);
    assert_close!(0.0, vid.sq_dist(&single, &[3.0, 4.0], 2));

    // Distance to origin: ||[3,4]-[0,0]||² * 1/2 = 25 * 0.5 = 12.5
    assert_close!(12.5, vid.sq_dist(&single, &[0.0, 0.0], 2));

    // Single-point CFs: ||c1-c2||² * n1*n2/(n1+n2) = ||c1-c2||² * 1/2
    let mut single2 = VII::<f64>::new(2);
    single2.add(&[6.0, 8.0]);
    // ||[3,4]-[6,8]||² = 9+16 = 25, * 1/2 = 12.5
    assert_close!(12.5, vid.sq_dist_cf(&single, &single2, 2));
  }

  #[test]
  fn test_variance_increase_various_dimensions() {
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

      let vid = VarianceIncreaseDistance::<f64, VII<f64>>::new();
      let expected = reference_variance_increase(
        &points.iter().map(|p| &p[..dim]).collect::<Vec<_>>(),
        &query[..dim],
        dim,
      );
      assert_close!(
        expected,
        vid.sq_dist(&cf, &query[..dim], dim),
        "dim={}",
        dim
      );
    }
  }
}
