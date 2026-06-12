use std::marker::PhantomData;

use crate::{cluster_feature::ClusterFeature, types::Float};

use super::{CFDistance, SqEuclidean, VectorDistance};

/// Average Radius distance (BIRCH "R" criterion).
///
/// Measures the average squared radius of the cluster that would result from
/// adding a point or merging two clusters. Unlike VarianceIncreaseDistance
/// which only considers the SSD increase, this accounts for the existing
/// spread of the cluster.
///
/// Formulas (derived from ELKI RadiusDistance):
///   sq_dist(CF, v)  = (n/(n+1) * ||v - centroid||² + ssd) / (n+1)
///   sq_dist(CF1, CF2) = (n1*n2/(n1+n2) * ||c1-c2||² + ssd1 + ssd2) / (n1+n2)
///
/// Equivalent to the new SSD after merging, divided by the new size.
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
/// T. Zhang, R. Ramakrishnan, M. Livny
/// "BIRCH: An Efficient Data Clustering Method for Very Large Databases"
/// Proc. 1996 ACM SIGMOD.
pub struct RadiusDistance<F: Float, C: ClusterFeature<F>> {
  _marker: PhantomData<C>,
  _marker2: PhantomData<F>,
}
impl<F: Float, C: ClusterFeature<F>> Clone for RadiusDistance<F, C> {
  fn clone(&self) -> Self {
    Self::new()
  }
}
impl<F: Float, C: ClusterFeature<F>> Default for RadiusDistance<F, C> {
  fn default() -> Self {
    Self::new()
  }
}

impl<F: Float, C: ClusterFeature<F>> RadiusDistance<F, C> {
  pub fn new() -> Self {
    RadiusDistance {
      _marker: PhantomData,
      _marker2: PhantomData,
    }
  }
}
impl<F: Float, C: ClusterFeature<F>> CFDistance<F, C> for RadiusDistance<F, C> {
  fn sq_dist(&self, cf1: &C, x: &[F], d: usize) -> F {
    let n = cf1.size();
    if n == 0 {
      return F::zero();
    }
    // (n/(n+1) * ||v - centroid||² + ssd) / (n+1)
    // Optimized: (n * centroid_dist + (n+1) * ssd) / (n+1)²
    // to avoid one division
    let centroid_dist = SqEuclidean::dist(cf1.centroid(), x, d);
    let n_f = F::from_index(n);
    let n_plus_1 = n_f + F::one();
    let numerator = n_f.mul_add(centroid_dist, n_plus_1 * cf1.ssd());
    numerator / (n_plus_1 * n_plus_1)
  }

  fn sq_dist_cf(&self, cf1: &C, cf2: &C, d: usize) -> F {
    let n1 = cf1.size();
    let n2 = cf2.size();
    let n12 = n1 + n2;
    if n12 == 0 {
      return F::zero();
    }
    // (n1*n2/(n1+n2) * ||c1-c2||² + ssd1 + ssd2) / (n1+n2)
    // Optimized: (n1*n2 * centroid_dist + n12*(ssd1+ssd2)) / n12²
    // to avoid one division
    let centroid_dist = SqEuclidean::dist(cf1.centroid(), cf2.centroid(), d);
    let n1_f = F::from_index(n1);
    let n2_f = F::from_index(n2);
    let n12_f = F::from_index(n12);
    let ssd_sum = cf1.ssd() + cf2.ssd();
    let numerator = n1_f.mul_add(n2_f * centroid_dist, n12_f * ssd_sum);
    numerator / (n12_f * n12_f)
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

  /// Reference: average radius after adding v = (n/(n+1)*||v-c||² + ssd) / (n+1)
  fn reference_radius(points: &[&[f64]], v: &[f64], d: usize) -> f64 {
    let n = points.len();
    if n == 0 {
      return 0.0;
    }
    // Compute centroid and ssd
    let mut centroid = vec![0.0; d];
    for p in points {
      for i in 0..d {
        centroid[i] += p[i];
      }
    }
    for i in 0..d {
      centroid[i] /= n as f64;
    }
    let mut ssd = 0.0;
    for p in points {
      for i in 0..d {
        let diff = p[i] - centroid[i];
        ssd += diff * diff;
      }
    }
    // Radius formula: (n/(n+1) * ||v-c||² + ssd) / (n+1)
    let mut dist_sq = 0.0;
    for i in 0..d {
      let diff = v[i] - centroid[i];
      dist_sq += diff * diff;
    }
    (n as f64 / (n + 1) as f64 * dist_sq + ssd) / (n + 1) as f64
  }

  /// Reference: average radius after merging = (n1*n2/(n1+n2)*||c1-c2||² + ssd1+ssd2) / (n1+n2)
  fn reference_radius_cf(points1: &[&[f64]], points2: &[&[f64]], d: usize) -> f64 {
    let n1 = points1.len();
    let n2 = points2.len();
    let n12 = n1 + n2;
    if n12 == 0 {
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
    let mut ssd1 = 0.0;
    let mut ssd2 = 0.0;
    for p in points1 {
      for i in 0..d {
        let diff = p[i] - c1[i];
        ssd1 += diff * diff;
      }
    }
    for p in points2 {
      for i in 0..d {
        let diff = p[i] - c2[i];
        ssd2 += diff * diff;
      }
    }
    let mut dist_sq = 0.0;
    for i in 0..d {
      let diff = c1[i] - c2[i];
      dist_sq += diff * diff;
    }
    (n1 as f64 * n2 as f64 / n12 as f64 * dist_sq + ssd1 + ssd2) / n12 as f64
  }

  /// Reference: verify by computing actual average radius after merge
  fn reference_radius_by_ssd(points: &[&[f64]], v: &[f64], d: usize) -> f64 {
    let n = points.len();
    if n == 0 {
      return 0.0;
    }
    // New centroid and SSD after adding v
    let mut new_centroid = vec![0.0; d];
    for p in points {
      for i in 0..d {
        new_centroid[i] += p[i];
      }
    }
    for i in 0..d {
      new_centroid[i] += v[i];
    }
    for i in 0..d {
      new_centroid[i] /= (n + 1) as f64;
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
    new_ssd / (n + 1) as f64
  }

  #[test]
  fn test_radius() {
    let rd = RadiusDistance::<f64, VII<f64>>::new();

    // CF with points: [1,2,3], [3,4,5], [5,6,7] -> centroid=[3,4,5], ssd=24
    let mut cf = VII::<f64>::new(3);
    cf.add(&[1.0, 2.0, 3.0]);
    cf.add(&[3.0, 4.0, 5.0]);
    cf.add(&[5.0, 6.0, 7.0]);

    let points: [f64; 9] = [1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 5.0, 6.0, 7.0];
    let rows: &[&[f64]] = &[&points[0..3], &points[3..6], &points[6..9]];

    // Query [0,0,0]: (3/4*50 + 24) / 4 = (37.5+24)/4 = 61.5/4 = 15.375
    let expected = reference_radius(rows, &[0.0, 0.0, 0.0], 3);
    assert_close!(
      expected,
      rd.sq_dist(&cf, &[0.0, 0.0, 0.0], 3),
      "query=[0,0,0], expected={}",
      expected
    );

    // Query centroid [3,4,5]: (3/4*0 + 24) / 4 = 24/4 = 6
    assert_close!(6.0, rd.sq_dist(&cf, &[3.0, 4.0, 5.0], 3));

    // Verify against SSD-based reference
    let ssd_ref = reference_radius_by_ssd(rows, &[0.0, 0.0, 0.0], 3);
    assert_close!(
      ssd_ref,
      rd.sq_dist(&cf, &[0.0, 0.0, 0.0], 3),
      "SSD reference={}",
      ssd_ref
    );
  }

  #[test]
  fn test_radius_cf_to_cf() {
    let rd = RadiusDistance::<f64, VII<f64>>::new();

    let mut cf1 = VII::<f64>::new(2);
    cf1.add(&[1.0, 2.0]);
    cf1.add(&[3.0, 4.0]);

    let mut cf2 = VII::<f64>::new(2);
    cf2.add(&[5.0, 6.0]);
    cf2.add(&[7.0, 8.0]);

    // ||c1-c2||² = 32, n1=n2=2, n1*n2/n12 = 1
    // ssd1 = 4, ssd2 = 4
    // (1*32 + 4 + 4) / 4 = 40/4 = 10
    let p1: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
    let p2: [f64; 4] = [5.0, 6.0, 7.0, 8.0];
    let rows1: &[&[f64]] = &[&p1[0..2], &p1[2..4]];
    let rows2: &[&[f64]] = &[&p2[0..2], &p2[2..4]];
    let expected = reference_radius_cf(rows1, rows2, 2);
    assert_close!(expected, rd.sq_dist_cf(&cf1, &cf2, 2));
  }

  #[test]
  fn test_radius_edge_cases() {
    let rd = RadiusDistance::<f64, VII<f64>>::new();

    // Empty CF returns 0
    let empty = VII::<f64>::new(2);
    assert_eq!(rd.sq_dist(&empty, &[1.0, 2.0], 2), 0.0);
    assert_eq!(rd.sq_dist_cf(&empty, &empty, 2), 0.0);

    // Single point CF: ssd=0, n=1
    // (1/2 * ||v-c||² + 0) / 2 = ||v-c||² / 4
    let mut single = VII::<f64>::new(2);
    single.add(&[3.0, 4.0]);
    // Distance to same point: 0/4 = 0
    assert_close!(0.0, rd.sq_dist(&single, &[3.0, 4.0], 2));
    // Distance to origin: 25/4 = 6.25
    assert_close!(6.25, rd.sq_dist(&single, &[0.0, 0.0], 2));

    // Two single-point CFs: n12 = 2
    // (1*1/2 * ||c1-c2||² + 0 + 0) / 2 = ||c1-c2||² / 4
    let mut single2 = VII::<f64>::new(2);
    single2.add(&[6.0, 8.0]);
    // ||[3,4]-[6,8]||² = 25, /4 = 6.25
    assert_close!(6.25, rd.sq_dist_cf(&single, &single2, 2));
  }

  #[test]
  fn test_radius_various_dimensions() {
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

      let rd = RadiusDistance::<f64, VII<f64>>::new();
      let expected = reference_radius(
        &points.iter().map(|p| &p[..dim]).collect::<Vec<_>>(),
        &query[..dim],
        dim,
      );
      assert_close!(expected, rd.sq_dist(&cf, &query[..dim], dim), "dim={}", dim);
    }
  }
}
