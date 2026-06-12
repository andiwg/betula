use super::ClusterFeature;
use crate::types::Float;

/// VII (Value, Index, Index) cluster feature.
///
/// Stores a single scalar sum of squared deviations (SSD) alongside the
/// centroid and count. This is the simplest and most common cluster feature
/// type, equivalent to storing the total variance without per-dimension detail.
///
/// Reference: ELKI BaseCF (VII feature)
pub struct VII<F: Copy> {
  size: usize,
  centroid: Vec<F>,
  ssd: F,
  dim: usize,
}
impl<F: Float> Clone for VII<F> {
  fn clone(&self) -> Self {
    VII {
      size: self.size,
      centroid: self.centroid.clone(),
      ssd: self.ssd,
      dim: self.dim,
    }
  }
}
impl<F: Float> ClusterFeature<F> for VII<F> {
  fn new(dim: usize) -> Self {
    VII {
      size: 0,
      centroid: vec![F::zero(); dim],
      ssd: F::zero(),
      dim,
    }
  }

  fn reset(&mut self) {
    self.size = 0;
    self.centroid.fill(F::zero());
    self.ssd = F::zero();
  }

  #[inline]
  fn size(&self) -> usize {
    self.size
  }

  #[inline]
  fn centroid(&self) -> &[F] {
    &self.centroid
  }

  #[inline]
  fn ssd(&self) -> F {
    self.ssd
  }

  /// Returns an isotropic approximation of per-dimension variance.
  /// VII stores only a scalar SSD, so the total is divided evenly.
  #[inline]
  fn variance(&self, _d: usize) -> F {
    if self.size < 2 {
      return F::zero();
    }
    let dim = F::from_index(self.dim);
    self.ssd / dim / F::from_index(self.size)
  }

  /// Returns `None` — VII does not store cross-product information.
  /// Use [`VVV`] if you need the covariance matrix.
  ///
  /// [`VVV`]: crate::cluster_feature::VVV
  fn covariance(&self) -> Option<Vec<Vec<F>>> {
    None
  }

  fn ssd_per_dim(&self) -> Option<&[F]> {
    // VII stores only a scalar SSD, not per-dimension.
    None
  }

  #[inline]
  fn ssd_upper(&self) -> Option<&[F]> {
    // VII does not store upper-triangular matrix.
    None
  }

  fn add(&mut self, x: &[F]) {
    if self.size == 0 {
      self.centroid.copy_from_slice(x);
      self.size = 1;
      self.ssd = F::zero();
    } else {
      self.size += 1;
      let n = F::from_index(self.size);
      // Welford update: ssd += delta * (x - new_mean) = delta^2 * (1 - 1/n)
      let one_minus_inv_n = F::one() - F::one() / n;
      for (c, v) in self.centroid.iter_mut().zip(x) {
        let delta = *v - *c;
        self.ssd += delta * delta * one_minus_inv_n;
        *c += delta / n;
      }
    }
  }

  fn add_cf<CF: ClusterFeature<F>>(&mut self, other: &CF) {
    if self.size == 0 {
      self.centroid.copy_from_slice(other.centroid());
      self.size = other.size();
      self.ssd = other.ssd();
    } else {
      let other_n = F::from_index(other.size());
      self.size += other.size();
      let combined_n = F::from_index(self.size);
      let factor = other_n / combined_n;
      self.ssd += other.ssd();
      // Parallel update: ssd += other_n * delta * (other_mean - new_mean)
      // = other_n * delta^2 * (1 - factor)
      let other_n_times_one_minus_factor = other_n * (F::one() - factor);
      for (c, o) in self.centroid.iter_mut().zip(other.centroid()) {
        let delta = *o - *c;
        self.ssd += other_n_times_one_minus_factor * delta * delta;
        *c += delta * factor;
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::cluster_feature::{VVI, VVV};

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

  #[test]
  fn test_vii_add_single_point() {
    let mut vii = VII::<f64>::new(3);
    vii.add(&[1.0, 2.0, 3.0]);

    assert_eq!(vii.size, 1);
    assert_close!(1.0, vii.centroid[0]);
    assert_close!(2.0, vii.centroid[1]);
    assert_close!(3.0, vii.centroid[2]);
    assert_close!(0.0, vii.ssd);
  }

  #[test]
  fn test_vii_add_multiple_points() {
    let mut vii = VII::<f64>::new(3);
    vii.add(&[1.0, 2.0, 3.0]);
    vii.add(&[3.0, 4.0, 5.0]);
    vii.add(&[5.0, 6.0, 7.0]);

    assert_eq!(vii.size, 3);
    // Centroid: [3, 4, 5]
    assert_close!(3.0, vii.centroid[0]);
    assert_close!(4.0, vii.centroid[1]);
    assert_close!(5.0, vii.centroid[2]);
    // Total SSD = (1-3)²+(2-4)²+(3-5)² + (3-3)²+(4-4)²+(5-5)² + (5-3)²+(6-4)²+(7-5)²
    //           = 4+4+4 + 0+0+0 + 4+4+4 = 24
    assert_close!(24.0, vii.ssd);
  }

  #[test]
  fn test_vii_variance_isotropic() {
    let mut vii = VII::<f64>::new(3);
    vii.add(&[1.0, 2.0, 3.0]);
    vii.add(&[3.0, 4.0, 5.0]);
    vii.add(&[5.0, 6.0, 7.0]);

    // variance = ssd / dim / n = 24 / 3 / 3 = 8/3
    assert_close!(8.0 / 3.0, vii.variance(0));
    assert_close!(8.0 / 3.0, vii.variance(1));
    assert_close!(8.0 / 3.0, vii.variance(2));
  }

  #[test]
  fn test_vii_add_cf_vii_to_vii() {
    let mut vii1 = VII::<f64>::new(2);
    vii1.add(&[1.0, 2.0]);
    vii1.add(&[3.0, 4.0]);

    let mut vii2 = VII::<f64>::new(2);
    vii2.add(&[5.0, 6.0]);
    vii2.add(&[7.0, 8.0]);

    // vii1: centroid=[2,3], ssd=4
    // vii2: centroid=[6,7], ssd=4
    assert_close!(2.0, vii1.centroid[0]);
    assert_close!(4.0, vii1.ssd);

    // Merge vii2 into vii1
    vii1.add_cf(&vii2);

    assert_eq!(vii1.size, 4);
    // New centroid: [4, 5]
    assert_close!(4.0, vii1.centroid[0]);
    assert_close!(5.0, vii1.centroid[1]);
    // Total SSD: [1,3,5,7] mean=4, SSD_0 = 9+1+1+9 = 20; [2,4,6,8] mean=5, SSD_1 = 20
    assert_close!(40.0, vii1.ssd);
  }

  #[test]
  fn test_vii_add_cf_empty() {
    let mut empty = VII::<f64>::new(3);

    let mut vii = VII::<f64>::new(3);
    vii.add(&[1.0, 2.0, 3.0]);
    vii.add(&[4.0, 5.0, 6.0]);

    empty.add_cf(&vii);

    assert_eq!(empty.size, 2);
    assert_close!(2.5, empty.centroid[0]);
    assert_close!(3.5, empty.centroid[1]);
    assert_close!(4.5, empty.centroid[2]);
    // SSD: (1-2.5)²+(4-2.5)² + (2-3.5)²+(5-3.5)² + (3-4.5)²+(6-4.5)² = 4.5*3 = 13.5
    assert_close!(13.5, empty.ssd);
  }

  #[test]
  fn test_vii_ssd_per_dim_none() {
    // VII stores only scalar SSD, not per-dimension.
    let mut vii = VII::<f64>::new(3);
    vii.add(&[1.0, 2.0, 3.0]);
    vii.add(&[3.0, 4.0, 5.0]);

    assert!(vii.ssd_per_dim().is_none());
  }

  #[test]
  fn test_vii_ssd_upper_none() {
    // VII does not store upper-triangular matrix, so ssd_upper returns None.
    let mut vii = VII::<f64>::new(3);
    vii.add(&[1.0, 2.0, 3.0]);
    vii.add(&[3.0, 4.0, 5.0]);

    assert!(vii.ssd_upper().is_none());
  }

  #[test]
  fn test_vii_reset() {
    let mut vii = VII::<f64>::new(3);
    vii.add(&[1.0, 2.0, 3.0]);
    vii.add(&[4.0, 5.0, 6.0]);

    vii.reset();

    assert_eq!(vii.size, 0);
    assert_close!(0.0, vii.centroid[0]);
    assert_close!(0.0, vii.ssd);
  }

  #[test]
  fn test_vii_covariance_none() {
    let vii = VII::<f64>::new(3);
    assert!(vii.covariance().is_none());
  }

  #[test]
  fn test_vii_variance_empty() {
    let vii = VII::<f64>::new(3);
    assert_close!(0.0, vii.variance(0));
    assert_close!(0.0, vii.variance(2));
  }

  #[test]
  fn test_vii_variance_single_point() {
    let mut vii = VII::<f64>::new(3);
    vii.add(&[1.0, 2.0, 3.0]);
    // Single point: ssd is 0, so variance is 0
    assert_close!(0.0, vii.variance(0));
    assert_close!(0.0, vii.variance(1));
    assert_close!(0.0, vii.variance(2));
  }

  #[test]
  fn test_vii_various_dimensions() {
    let dims = [1, 2, 5, 8, 16];
    for &dim in &dims {
      let mut vii = VII::<f64>::new(dim);
      for i in 0..10 {
        let point: Vec<f64> = (0..dim).map(|j| (i * dim + j) as f64).collect();
        vii.add(&point);
      }
      assert_eq!(vii.size, 10);
      assert!(vii.ssd >= 0.0, "ssd < 0 for dim {}", dim);
    }
  }

  #[test]
  fn test_vii_vs_vvi_total_ssd() {
    let points: Vec<Vec<f64>> = vec![
      vec![1.0, 2.0, 3.0],
      vec![3.0, 4.0, 5.0],
      vec![5.0, 6.0, 7.0],
      vec![2.0, 8.0, 1.0],
      vec![4.0, 0.0, 9.0],
    ];

    let mut vii = VII::<f64>::new(3);
    let mut vvi = VVI::<f64>::new(3);
    for p in &points {
      vii.add(p);
      vvi.add(p);
    }

    assert_close!(vvi.ssd(), vii.ssd);
  }

  #[test]
  fn test_vii_vs_vvv_total_ssd() {
    let points: Vec<Vec<f64>> = vec![
      vec![1.0, 2.0, 3.0],
      vec![3.0, 4.0, 5.0],
      vec![5.0, 6.0, 7.0],
      vec![2.0, 8.0, 1.0],
      vec![4.0, 0.0, 9.0],
    ];

    let mut vii = VII::<f64>::new(3);
    let mut vvv = VVV::<f64>::new(3);
    for p in &points {
      vii.add(p);
      vvv.add(p);
    }

    assert_close!(vvv.ssd(), vii.ssd);
  }

  #[test]
  fn test_vii_add_cf_vii_to_vvi() {
    let mut vvi = VVI::<f64>::new(3);
    vvi.add(&[1.0, 2.0, 3.0]);
    vvi.add(&[3.0, 4.0, 5.0]);

    let mut vii = VII::<f64>::new(3);
    vii.add(&[5.0, 6.0, 7.0]);
    vii.add(&[7.0, 8.0, 9.0]);

    vvi.add_cf(&vii);

    assert_eq!(vvi.size(), 4);
    // Centroid: [4, 5, 6]
    assert_close!(4.0, vvi.centroid()[0]);
    assert_close!(5.0, vvi.centroid()[1]);
    assert_close!(6.0, vvi.centroid()[2]);

    let mut ref_vii = VII::<f64>::new(3);
    for p in &[
      vec![1.0, 2.0, 3.0],
      vec![3.0, 4.0, 5.0],
      vec![5.0, 6.0, 7.0],
      vec![7.0, 8.0, 9.0],
    ] {
      ref_vii.add(p);
    }
    assert_close!(ref_vii.ssd, vvi.ssd());
  }

  #[test]
  fn test_vii_add_cf_vii_to_vvv() {
    let mut vvv = VVV::<f64>::new(3);
    vvv.add(&[1.0, 2.0, 3.0]);
    vvv.add(&[3.0, 4.0, 5.0]);

    let mut vii = VII::<f64>::new(3);
    vii.add(&[5.0, 6.0, 7.0]);
    vii.add(&[7.0, 8.0, 9.0]);

    vvv.add_cf(&vii);

    assert_eq!(vvv.size(), 4);
    assert_close!(4.0_f64, vvv.centroid()[0]);
    assert_close!(5.0_f64, vvv.centroid()[1]);
    assert_close!(6.0_f64, vvv.centroid()[2]);

    let mut ref_vii = VII::<f64>::new(3);
    for p in &[
      vec![1.0, 2.0, 3.0],
      vec![3.0, 4.0, 5.0],
      vec![5.0, 6.0, 7.0],
      vec![7.0, 8.0, 9.0],
    ] {
      ref_vii.add(p);
    }
    assert_close!(ref_vii.ssd, vvv.ssd());
  }

  #[test]
  fn test_vii_add_cf_vvi_to_vii() {
    let mut vii = VII::<f64>::new(3);
    vii.add(&[1.0, 2.0, 3.0]);
    vii.add(&[3.0, 4.0, 5.0]);

    let mut vvi = VVI::<f64>::new(3);
    vvi.add(&[5.0, 6.0, 7.0]);
    vvi.add(&[7.0, 8.0, 9.0]);

    vii.add_cf(&vvi);

    assert_eq!(vii.size, 4);
    assert_close!(4.0, vii.centroid[0]);
    assert_close!(5.0, vii.centroid[1]);
    assert_close!(6.0, vii.centroid[2]);

    let mut ref_vii = VII::<f64>::new(3);
    for p in &[
      vec![1.0, 2.0, 3.0],
      vec![3.0, 4.0, 5.0],
      vec![5.0, 6.0, 7.0],
      vec![7.0, 8.0, 9.0],
    ] {
      ref_vii.add(p);
    }
    assert_close!(ref_vii.ssd, vii.ssd);
  }

  #[test]
  fn test_vii_add_cf_vvv_to_vii() {
    let mut vii = VII::<f64>::new(3);
    vii.add(&[1.0, 2.0, 3.0]);
    vii.add(&[3.0, 4.0, 5.0]);

    let mut vvv = VVV::<f64>::new(3);
    vvv.add(&[5.0, 6.0, 7.0]);
    vvv.add(&[7.0, 8.0, 9.0]);

    vii.add_cf(&vvv);

    assert_eq!(vii.size, 4);
    assert_close!(4.0, vii.centroid[0]);
    assert_close!(5.0, vii.centroid[1]);
    assert_close!(6.0, vii.centroid[2]);

    let mut ref_vii = VII::<f64>::new(3);
    for p in &[
      vec![1.0, 2.0, 3.0],
      vec![3.0, 4.0, 5.0],
      vec![5.0, 6.0, 7.0],
      vec![7.0, 8.0, 9.0],
    ] {
      ref_vii.add(p);
    }
    assert_close!(ref_vii.ssd, vii.ssd);
  }

  #[test]
  fn test_vii_add_cf_empty_other() {
    let mut vii = VII::<f64>::new(2);
    vii.add(&[1.0, 2.0]);
    vii.add(&[4.0, 1.0]);

    let empty = VII::<f64>::new(2);

    let saved_size = vii.size;
    let saved_centroid = vii.centroid.clone();
    let saved_ssd = vii.ssd;

    vii.add_cf(&empty);

    assert_eq!(vii.size, saved_size);
    assert_close!(saved_centroid[0], vii.centroid[0]);
    assert_close!(saved_ssd, vii.ssd);
  }

  #[test]
  fn test_vii_clone() {
    let mut vii = VII::<f64>::new(2);
    vii.add(&[1.0, 2.0]);
    vii.add(&[4.0, 1.0]);

    let vii2 = vii.clone();

    assert_eq!(vii.size, vii2.size);
    assert_close!(vii.centroid[0], vii2.centroid[0]);
    assert_close!(vii.ssd, vii2.ssd);

    // Mutate original, clone should be unaffected
    vii.add(&[5.0, 5.0]);
    assert_eq!(vii2.size, 2);
    assert_close!(2.5, vii2.centroid[0]);
  }
}
