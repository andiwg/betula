use super::ClusterFeature;
use crate::types::Float;

/// VVI (Variance per dimension) cluster feature.
///
/// Unlike VII which stores a single scalar SSD, VVI stores
/// per-dimension sum of squared deviations. This allows computing exact
/// per-dimension variance as `ssd[d] / n` instead of the isotropic
/// approximation `ssd / dim / n`.
///
/// Reference: ELKI VVIFeature
pub struct VVI<F> {
  size: usize,
  centroid: Vec<F>,
  ssd: Vec<F>,
  dim: usize,
}

impl<F: Float> Clone for VVI<F> {
  fn clone(&self) -> Self {
    VVI {
      size: self.size,
      centroid: self.centroid.clone(),
      ssd: self.ssd.clone(),
      dim: self.dim,
    }
  }
}

impl<F: Float> ClusterFeature<F> for VVI<F> {
  fn new(dim: usize) -> Self {
    VVI {
      size: 0,
      centroid: vec![F::zero(); dim],
      ssd: vec![F::zero(); dim],
      dim,
    }
  }

  fn reset(&mut self) {
    self.size = 0;
    self.centroid.fill(F::zero());
    self.ssd.fill(F::zero());
  }

  #[inline]
  fn size(&self) -> usize {
    self.size
  }

  #[inline]
  fn centroid(&self) -> &[F] {
    &self.centroid
  }

  /// Returns the total SSD (sum of all per-dimension SSDs).
  /// Matches the Java VVIFeature.sumdev() behavior.
  #[inline]
  fn ssd(&self) -> F {
    self.ssd.iter().copied().sum()
  }

  /// Returns the exact per-dimension variance: ssd[d] / n.
  /// Returns 0.0 for negative results (numerical safety).
  #[inline]
  fn variance(&self, d: usize) -> F {
    if self.size == 0 {
      return F::zero();
    }
    let var = self.ssd[d] / F::from_index(self.size);
    if var >= F::zero() { var } else { F::zero() }
  }

  /// Returns `None` — VVI does not store cross-product information.
  /// Use [`VVV`] if you need the covariance matrix.
  ///
  /// [`VVV`]: crate::cluster_feature::VVV
  fn covariance(&self) -> Option<Vec<Vec<F>>> {
    None
  }

  fn add(&mut self, x: &[F]) {
    if self.size == 0 {
      self.centroid[..self.dim].copy_from_slice(&x[..self.dim]);
      self.ssd.fill(F::zero());
      self.size = 1;
    } else {
      self.size += 1;
      let n = F::from_index(self.size);
      // Welford update: ssd[i] += delta * (x[i] - new_mean[i]) = delta^2 * (1 - 1/n)
      let one_minus_inv_n = F::one() - F::one() / n;
      for (i, (ci, xi)) in self.centroid.iter_mut().zip(x).enumerate() {
        let delta = *xi - *ci;
        self.ssd[i] += delta * delta * one_minus_inv_n;
        *ci += delta / n;
      }
    }
  }

  fn add_cf<CF: ClusterFeature<F>>(&mut self, other: &CF) {
    if self.size == 0 {
      self.centroid.copy_from_slice(other.centroid());
      self.size = other.size();
      // Try to get per-dimension SSD, compute from scalar if not available
      if let Some(per_dim) = other.ssd_per_dim() {
        self.ssd.copy_from_slice(per_dim);
      } else {
        // Compute per-dimension SSD from scalar (e.g., for VII: ssd / dim)
        let per_dim_val = other.ssd() / F::from_index(self.dim);
        self.ssd.fill(per_dim_val);
      }
    } else {
      let other_n = F::from_index(other.size());
      self.size += other.size();
      let combined_n = F::from_index(self.size);
      let factor = other_n / combined_n;

      // Get per-dimension SSD if available, otherwise use scalar approximation
      let other_per_dim_slice = other.ssd_per_dim();
      let other_n_times_one_minus_factor = other_n * (F::one() - factor);

      for (i, (ci, oi)) in self.centroid.iter_mut().zip(other.centroid()).enumerate() {
        let delta = *oi - *ci;
        let other_ssd_i = other_per_dim_slice.map_or_else(
          || other.ssd() / F::from_index(self.dim),
          |per_dim| per_dim[i],
        );
        self.ssd[i] += other_ssd_i + other_n_times_one_minus_factor * delta * delta;
        *ci += delta * factor;
      }
    }
  }

  #[inline]
  fn ssd_per_dim(&self) -> Option<&[F]> {
    Some(&self.ssd)
  }

  #[inline]
  fn ssd_upper(&self) -> Option<&[F]> {
    // VVI stores only diagonal, not upper-triangular matrix.
    None
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::cluster_feature::VII;

  macro_rules! assert_close {
        ($a:expr, $b:expr) => {
            assert!(
                ($a - $b).abs() < 1e-10,
                "expected={}, actual={}",
                $b,
                $a
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
  fn test_vvi_add_single_point() {
    let mut vvi = VVI::<f64>::new(3);
    vvi.add(&[1.0, 2.0, 3.0]);

    assert_eq!(vvi.size, 1);
    assert_close!(1.0, vvi.centroid[0]);
    assert_close!(2.0, vvi.centroid[1]);
    assert_close!(3.0, vvi.centroid[2]);
    assert_close!(0.0, vvi.ssd[0]);
    assert_close!(0.0, vvi.ssd[1]);
    assert_close!(0.0, vvi.ssd[2]);
    assert_close!(0.0, vvi.ssd()); // total SSD
  }

  #[test]
  fn test_vvi_add_multiple_points() {
    let mut vvi = VVI::<f64>::new(3);
    vvi.add(&[1.0, 2.0, 3.0]);
    vvi.add(&[3.0, 4.0, 5.0]);
    vvi.add(&[5.0, 6.0, 7.0]);

    assert_eq!(vvi.size, 3);
    // Centroid: [3, 4, 5]
    assert_close!(3.0, vvi.centroid[0]);
    assert_close!(4.0, vvi.centroid[1]);
    assert_close!(5.0, vvi.centroid[2]);

    // Per-dimension SSD: each dim has values [1,3,5], mean=3, SSD = (1-3)²+(3-3)²+(5-3)² = 4+0+4 = 8
    assert_close!(8.0, vvi.ssd[0]);
    assert_close!(8.0, vvi.ssd[1]);
    assert_close!(8.0, vvi.ssd[2]);
    assert_close!(24.0, vvi.ssd()); // total SSD = 8+8+8
  }

  #[test]
  fn test_vvi_variance_per_dimension() {
    let mut vvi = VVI::<f64>::new(3);
    vvi.add(&[1.0, 2.0, 3.0]);
    vvi.add(&[3.0, 4.0, 5.0]);
    vvi.add(&[5.0, 6.0, 7.0]);

    // variance(d) = ssd[d] / n = 8 / 3
    assert_close!(8.0 / 3.0, vvi.variance(0));
    assert_close!(8.0 / 3.0, vvi.variance(1));
    assert_close!(8.0 / 3.0, vvi.variance(2));
  }

  #[test]
  fn test_vvi_variance_asymmetric() {
    // Different variance per dimension
    let mut vvi = VVI::<f64>::new(2);
    vvi.add(&[0.0, 0.0]);
    vvi.add(&[10.0, 1.0]);
    vvi.add(&[20.0, 2.0]);

    // Dim 0: [0,10,20], mean=10, SSD = 100+0+100 = 200
    // Dim 1: [0,1,2], mean=1, SSD = 1+0+1 = 2
    assert_close!(200.0, vvi.ssd[0]);
    assert_close!(2.0, vvi.ssd[1]);

    // variance(0) = 200/3, variance(1) = 2/3
    assert_close!(200.0 / 3.0, vvi.variance(0));
    assert_close!(2.0 / 3.0, vvi.variance(1));
  }

  #[test]
  fn test_vvi_add_cf_vvi_to_vvi() {
    let mut vvi1 = VVI::<f64>::new(2);
    vvi1.add(&[1.0, 2.0]);
    vvi1.add(&[3.0, 4.0]);

    let mut vvi2 = VVI::<f64>::new(2);
    vvi2.add(&[5.0, 6.0]);
    vvi2.add(&[7.0, 8.0]);

    // vvi1: centroid=[2,3], ssd=[2,2]
    // vvi2: centroid=[6,7], ssd=[2,2]
    assert_close!(2.0, vvi1.centroid[0]);
    assert_close!(2.0, vvi1.ssd[0]);

    // Merge vvi2 into vvi1
    vvi1.add_cf(&vvi2);

    assert_eq!(vvi1.size, 4);
    // New centroid: [4, 5]
    assert_close!(4.0, vvi1.centroid[0]);
    assert_close!(5.0, vvi1.centroid[1]);

    // Verify total SSD matches what VII would compute
    let mut base = VII::<f64>::new(2);
    base.add(&[1.0, 2.0]);
    base.add(&[3.0, 4.0]);
    base.add(&[5.0, 6.0]);
    base.add(&[7.0, 8.0]);
    assert_close!(base.ssd(), vvi1.ssd());

    // Per-dimension SSDs should be preserved (not just total)
    // Dim 0: [1,3,5,7], mean=4, SSD = 9+1+1+9 = 20
    // Dim 1: [2,4,6,8], mean=5, SSD = 9+1+1+9 = 20
    assert_close!(20.0, vvi1.ssd[0]);
    assert_close!(20.0, vvi1.ssd[1]);
  }

  #[test]
  fn test_vvi_add_cf_empty() {
    let mut empty = VVI::<f64>::new(3);

    let mut vvi = VVI::<f64>::new(3);
    vvi.add(&[1.0, 2.0, 3.0]);
    vvi.add(&[4.0, 5.0, 6.0]);

    empty.add_cf(&vvi);

    assert_eq!(empty.size, 2);
    assert_close!(2.5, empty.centroid[0]);
    assert_close!(3.5, empty.centroid[1]);
    assert_close!(4.5, empty.centroid[2]);
    // SSD per dim: dim0: (1-2.5)²+(4-2.5)² = 2.25+2.25 = 4.5
    assert_close!(4.5, empty.ssd[0]);
    assert_close!(4.5, empty.ssd[1]);
    assert_close!(4.5, empty.ssd[2]);
  }

  #[test]
  fn test_vvi_add_cf_vvi_to_vii() {
    // VII.add_cf should work with VVI as the "other"
    let mut base = VII::<f64>::new(3);
    base.add(&[1.0, 2.0, 3.0]);
    base.add(&[3.0, 4.0, 5.0]);

    let mut vvi = VVI::<f64>::new(3);
    vvi.add(&[5.0, 6.0, 7.0]);
    vvi.add(&[7.0, 8.0, 9.0]);

    base.add_cf(&vvi);

    assert_eq!(base.size(), 4);
    // Centroid: [4, 5, 6]
    assert_close!(4.0, base.centroid()[0]);
    assert_close!(5.0, base.centroid()[1]);
    assert_close!(6.0, base.centroid()[2]);
    // Total SSD should match
    let mut ref_cf = VII::<f64>::new(3);
    ref_cf.add(&[1.0, 2.0, 3.0]);
    ref_cf.add(&[3.0, 4.0, 5.0]);
    ref_cf.add(&[5.0, 6.0, 7.0]);
    ref_cf.add(&[7.0, 8.0, 9.0]);
    assert_close!(ref_cf.ssd(), base.ssd());
  }

  #[test]
  fn test_vvi_ssd_per_dim() {
    let mut vvi = VVI::<f64>::new(3);
    vvi.add(&[1.0, 2.0, 3.0]);
    vvi.add(&[3.0, 4.0, 5.0]);

    let per_dim = vvi.ssd_per_dim().unwrap();
    assert_eq!(per_dim.len(), 3);
    assert_close!(2.0, per_dim[0]);
    assert_close!(2.0, per_dim[1]);
    assert_close!(2.0, per_dim[2]);
  }

  #[test]
  fn test_vvi_reset() {
    let mut vvi = VVI::<f64>::new(3);
    vvi.add(&[1.0, 2.0, 3.0]);
    vvi.add(&[4.0, 5.0, 6.0]);

    vvi.reset();

    assert_eq!(vvi.size, 0);
    assert_close!(0.0, vvi.centroid[0]);
    assert_close!(0.0, vvi.ssd[0]);
    assert_close!(0.0, vvi.ssd());
  }

  #[test]
  fn test_vvi_covariance_none() {
    let vvi = VVI::<f64>::new(2);
    assert!(vvi.covariance().is_none());
  }

  #[test]
  fn test_vvi_variance_empty() {
    let vvi = VVI::<f64>::new(3);
    assert_close!(0.0, vvi.variance(0));
    assert_close!(0.0, vvi.variance(2));
  }

  #[test]
  fn test_vvi_variance_single_point() {
    let mut vvi = VVI::<f64>::new(3);
    vvi.add(&[1.0, 2.0, 3.0]);
    // Single point: ssd is 0, so variance is 0
    assert_close!(0.0, vvi.variance(0));
    assert_close!(0.0, vvi.variance(1));
    assert_close!(0.0, vvi.variance(2));
  }

  #[test]
  fn test_vvi_various_dimensions() {
    let dims = [1, 2, 5, 8, 16];
    for &dim in &dims {
      let mut vvi = VVI::<f64>::new(dim);
      for i in 0..10 {
        let point: Vec<f64> = (0..dim).map(|j| (i * dim + j) as f64).collect();
        vvi.add(&point);
      }
      assert_eq!(vvi.size, 10);
      // Total SSD should be non-negative
      assert!(vvi.ssd() >= 0.0);
      // Each per-dimension SSD should be non-negative
      for i in 0..dim {
        assert!(vvi.ssd[i] >= 0.0, "ssd[{}] < 0", i);
      }
    }
  }

  #[test]
  fn test_vvi_vs_vii_total_ssd() {
    // VVI total SSD should always match VII total SSD for the same data
    let points: Vec<Vec<f64>> = vec![
      vec![1.0, 2.0, 3.0],
      vec![3.0, 4.0, 5.0],
      vec![5.0, 6.0, 7.0],
      vec![2.0, 8.0, 1.0],
      vec![4.0, 0.0, 9.0],
    ];

    let mut vvi = VVI::<f64>::new(3);
    let mut base = VII::<f64>::new(3);
    for p in &points {
      vvi.add(p);
      base.add(p);
    }

    assert_close!(base.ssd(), vvi.ssd());
  }

  #[test]
  fn test_vvi_add_cf_preserves_per_dim_ssd() {
    // Key test: merging VVI with VVI preserves per-dimension SSDs exactly
    // (not just total SSD)
    let mut vvi1 = VVI::<f64>::new(2);
    // Dim 0 has high variance, dim 1 has low variance
    vvi1.add(&[0.0, 5.0]);
    vvi1.add(&[100.0, 5.0]);
    vvi1.add(&[200.0, 5.0]);

    let mut vvi2 = VVI::<f64>::new(2);
    vvi2.add(&[0.0, 5.0]);
    vvi2.add(&[100.0, 5.0]);
    vvi2.add(&[200.0, 5.0]);

    vvi1.add_cf(&vvi2);

    // Dim 0: [0,100,200,0,100,200], mean=100, SSD = 10000+0+10000+10000+0+10000 = 40000
    // Dim 1: [5,5,5,5,5,5], mean=5, SSD = 0
    assert_close!(40000.0, vvi1.ssd[0], "dim 0 SSD");
    assert_close!(0.0, vvi1.ssd[1], "dim 1 SSD");
    assert_close!(40000.0, vvi1.ssd(), "total SSD");
  }
}
