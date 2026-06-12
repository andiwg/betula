use super::ClusterFeature;
use crate::types::Float;

/// VVV (Variance-Variance-Variance) cluster feature.
///
/// Unlike VII which stores a single scalar SSD, or VVI which stores
/// per-dimension SSDs, VVV stores the full cross-product sum of squared
/// deviations matrix. This allows computing the full covariance matrix,
/// not just per-dimension variance.
///
/// Storage: `ssd_upper` is a flat `Vec<F>` holding the upper triangular
/// (including diagonal) elements in row-major order. Row `i` contributes
/// `dim - i` entries for `(i, i), (i, i+1), …, (i, dim-1)`.
///
/// Flat index of `(i, j)` where `j >= i`:
/// `idx = i * dim - i * (i - 1) / 2 + (j - i)`
///
/// Reference: ELKI VVVFeature
pub struct VVV<F> {
  size: usize,
  centroid: Vec<F>,
  /// Flat upper-triangular storage (including diagonal), row-major.
  ssd_upper: Vec<F>,
  dim: usize,
}

impl<F: Float> VVV<F> {
  /// Flat index for upper-triangular element `(i, j)` where `j >= i`.
  #[inline(always)]
  fn flat_idx(&self, i: usize, j: usize) -> usize {
    i * self.dim - i.saturating_sub(1) * i / 2 + (j - i)
  }

  /// Access SSD element symmetrically from flat upper-triangular storage.
  /// Uses branchless min/max to avoid conditional branching.
  #[inline(always)]
  fn get_ssd(&self, i: usize, j: usize) -> F {
    let lo = i.min(j);
    let hi = i.max(j);
    self.ssd_upper[self.flat_idx(lo, hi)]
  }

  /// Set SSD element in flat upper-triangular storage.
  /// Uses branchless min/max to avoid conditional branching.
  #[inline(always)]
  fn set_ssd(&mut self, i: usize, j: usize, val: F) {
    let lo = i.min(j);
    let hi = i.max(j);
    let idx = self.flat_idx(lo, hi);
    self.ssd_upper[idx] = val;
  }
}

impl<F: Float> Clone for VVV<F> {
  fn clone(&self) -> Self {
    VVV {
      size: self.size,
      centroid: self.centroid.clone(),
      ssd_upper: self.ssd_upper.clone(),
      dim: self.dim,
    }
  }
}

impl<F: Float> ClusterFeature<F> for VVV<F> {
  fn new(dim: usize) -> Self {
    VVV {
      size: 0,
      centroid: vec![F::zero(); dim],
      ssd_upper: vec![F::zero(); dim * (dim + 1) / 2],
      dim,
    }
  }

  fn reset(&mut self) {
    self.size = 0;
    self.centroid.fill(F::zero());
    self.ssd_upper.fill(F::zero());
  }

  fn size(&self) -> usize {
    self.size
  }

  fn centroid(&self) -> &[F] {
    &self.centroid
  }

  /// Returns the total SSD (trace of the cross-product matrix).
  /// Matches the Java VVVFeature.sumdev() behavior.
  #[inline]
  fn ssd(&self) -> F {
    (0..self.dim).map(|i| self.get_ssd(i, i)).sum()
  }

  /// Returns the per-dimension variance: ssd[d][d] / n.
  /// Returns 0.0 for empty features or negative results (numerical safety).
  #[inline]
  fn variance(&self, d: usize) -> F {
    if self.size == 0 {
      return F::zero();
    }
    let var = self.get_ssd(d, d) / F::from_index(self.size);
    if var >= F::zero() { var } else { F::zero() }
  }

  fn ssd_per_dim(&self) -> Option<&[F]> {
    // VVV stores cross-products, not per-dimension SSD.
    None
  }

  #[inline]
  fn ssd_upper(&self) -> Option<&[F]> {
    Some(&self.ssd_upper)
  }

  /// Returns the full covariance matrix: ssd[i][j] / n.
  fn covariance(&self) -> Option<Vec<Vec<F>>> {
    if self.size == 0 {
      return Some(vec![vec![F::zero(); self.dim]; self.dim]);
    }
    let f = F::one() / F::from_index(self.size);
    let mut cov = vec![vec![F::zero(); self.dim]; self.dim];
    for (i, row) in cov.iter_mut().enumerate() {
      for (j, cell) in row.iter_mut().enumerate() {
        *cell = self.get_ssd(i, j) * f;
      }
    }
    Some(cov)
  }

  fn add(&mut self, x: &[F]) {
    if self.size == 0 {
      self.centroid[..self.dim].copy_from_slice(&x[..self.dim]);
      // ssd_upper already zeroed
      self.size = 1;
      return;
    }

    let d = self.dim;
    let n = F::from_index(self.size);
    let f = F::one() / (n + F::one());

    // Compute deltas once, reuse for SSD update and centroid update.
    let deltas: Vec<F> = (0..d).map(|i| x[i] - self.centroid[i]).collect();

    // Welford update: ssd[i][j] += delta[i] * (x[j] - new_mean[j])
    // = delta[i] * delta[j] * (1 - f)
    let one_minus_f = F::one() - f;
    for (i, &delta_i) in deltas.iter().enumerate() {
      for (j, &delta_j) in deltas.iter().enumerate().take(i + 1) {
        self.set_ssd(i, j, self.get_ssd(i, j) + delta_i * delta_j * one_minus_f);
      }
    }

    // Update centroid
    for (i, delta_i) in deltas.iter().enumerate() {
      self.centroid[i] += *delta_i * f;
    }
    self.size += 1;
  }

  fn add_cf<CF: ClusterFeature<F>>(&mut self, other: &CF) {
    if self.size == 0 {
      self.centroid.copy_from_slice(other.centroid());
      self.size = other.size();
      // Build upper-triangular matrix from ssd_upper if available,
      // otherwise from ssd_per_dim (diagonal-only for VII/VVI).
      if let Some(mat) = other.ssd_upper() {
        for (i, _) in (0..self.dim).enumerate() {
          for j in 0..=i {
            let other_idx = j * self.dim - j.saturating_sub(1) * self.dim / 2 + (i - j);
            self.set_ssd(i, j, mat[other_idx]);
          }
        }
      } else if let Some(other_per_dim) = other.ssd_per_dim() {
        for (i, &val) in other_per_dim.iter().enumerate().take(self.dim) {
          self.set_ssd(i, i, val);
        }
      } else {
        // VII case: only scalar SSD available, distribute isotropically.
        let per_dim = other.ssd() / F::from_index(self.dim);
        for i in 0..self.dim {
          self.set_ssd(i, i, per_dim);
        }
      }
      return;
    }

    if other.size() == 0 {
      // Nothing to merge
      return;
    }

    let d = self.dim;
    let other_n = F::from_index(other.size());
    self.size += other.size();
    let combined_n = F::from_index(self.size);
    let factor = other_n / combined_n;

    // Compute deltas once, reuse for SSD update and centroid update.
    let deltas: Vec<F> = (0..d)
      .map(|i| other.centroid()[i] - self.centroid[i])
      .collect();

    let other_per_dim = other.ssd_per_dim();
    let other_mat = other.ssd_upper();
    let one_minus_factor = F::one() - factor;

    // Parallel update: ssd[i][j] += other_mat[i][j] + other_n * delta[i] * delta[j] * (1 - factor)
    for (i, &delta_i) in deltas.iter().enumerate() {
      for (j, &delta_j) in deltas.iter().enumerate().take(i + 1) {
        let other_val = if let Some(mat) = other_mat {
          let other_idx = j * d - j.saturating_sub(1) * d / 2 + (i - j);
          mat[other_idx]
        } else if i == j {
          other_per_dim.map_or_else(
            || other.ssd() / F::from_index(self.dim),
            |per_dim| per_dim[i],
          )
        } else {
          F::zero()
        };
        let val = self.get_ssd(i, j) + other_val + other_n * delta_i * delta_j * one_minus_factor;
        self.set_ssd(i, j, val);
      }
    }

    // Update centroid
    for (i, delta_i) in deltas.iter().enumerate() {
      self.centroid[i] += *delta_i * factor;
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::cluster_feature::{VII, VVI};

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
  fn test_vvv_add_single_point() {
    let mut vvv = VVV::<f64>::new(3);
    vvv.add(&[1.0, 2.0, 3.0]);

    assert_eq!(vvv.size, 1);
    assert_close!(1.0, vvv.centroid[0]);
    assert_close!(2.0, vvv.centroid[1]);
    assert_close!(3.0, vvv.centroid[2]);
    // Single point: all SSDs are 0
    assert_close!(0.0, vvv.get_ssd(0, 0));
    assert_close!(0.0, vvv.get_ssd(1, 1));
    assert_close!(0.0, vvv.get_ssd(2, 2));
    assert_close!(0.0, vvv.get_ssd(0, 1));
    assert_close!(0.0, vvv.get_ssd(0, 2));
    assert_close!(0.0, vvv.get_ssd(1, 2));
    assert_close!(0.0, vvv.ssd()); // total SSD (trace)
  }

  #[test]
  fn test_vvv_add_two_points() {
    let mut vvv = VVV::<f64>::new(2);
    vvv.add(&[1.0, 2.0]);
    vvv.add(&[4.0, 1.0]);

    assert_eq!(vvv.size, 2);
    // Centroid: [2.5, 1.5]
    assert_close!(2.5, vvv.centroid[0]);
    assert_close!(1.5, vvv.centroid[1]);

    // SSD[0][0] = (1-2.5)²+(4-2.5)² = 2.25+2.25 = 4.5
    // SSD[1][1] = (2-1.5)²+(1-1.5)² = 0.25+0.25 = 0.5
    // SSD[0][1] = (1-2.5)*(2-1.5)+(4-2.5)*(1-1.5) = -0.75-0.75 = -1.5
    assert_close!(4.5, vvv.get_ssd(0, 0));
    assert_close!(0.5, vvv.get_ssd(1, 1));
    assert_close!(-1.5, vvv.get_ssd(0, 1));
    assert_close!(-1.5, vvv.get_ssd(1, 0)); // symmetric
    assert_close!(5.0, vvv.ssd()); // trace = 4.5 + 0.5
  }

  #[test]
  fn test_vvv_add_multiple_points() {
    let mut vvv = VVV::<f64>::new(3);
    vvv.add(&[1.0, 2.0, 3.0]);
    vvv.add(&[3.0, 4.0, 5.0]);
    vvv.add(&[5.0, 6.0, 7.0]);

    assert_eq!(vvv.size, 3);
    // Centroid: [3, 4, 5]
    assert_close!(3.0, vvv.centroid[0]);
    assert_close!(4.0, vvv.centroid[1]);
    assert_close!(5.0, vvv.centroid[2]);

    // Each dim has values [1,3,5], mean=3, SSD = 4+0+4 = 8
    assert_close!(8.0, vvv.get_ssd(0, 0));
    assert_close!(8.0, vvv.get_ssd(1, 1));
    assert_close!(8.0, vvv.get_ssd(2, 2));

    // Cross-products: all dims move together, so cross = same as diagonal
    assert_close!(8.0, vvv.get_ssd(0, 1));
    assert_close!(8.0, vvv.get_ssd(0, 2));
    assert_close!(8.0, vvv.get_ssd(1, 2));

    assert_close!(24.0, vvv.ssd()); // trace = 8+8+8
  }

  #[test]
  fn test_vvv_variance_per_dimension() {
    let mut vvv = VVV::<f64>::new(3);
    vvv.add(&[1.0, 2.0, 3.0]);
    vvv.add(&[3.0, 4.0, 5.0]);
    vvv.add(&[5.0, 6.0, 7.0]);

    // variance(d) = ssd[d][d] / n = 8 / 3
    assert_close!(8.0 / 3.0, vvv.variance(0));
    assert_close!(8.0 / 3.0, vvv.variance(1));
    assert_close!(8.0 / 3.0, vvv.variance(2));
  }

  #[test]
  fn test_vvv_covariance_matrix() {
    let mut vvv = VVV::<f64>::new(2);
    vvv.add(&[1.0, 2.0]);
    vvv.add(&[4.0, 1.0]);

    let cov = vvv.covariance().unwrap();
    // cov[0][0] = 4.5/2 = 2.25
    // cov[1][1] = 0.5/2 = 0.25
    // cov[0][1] = -1.5/2 = -0.75
    assert_close!(2.25, cov[0][0]);
    assert_close!(0.25, cov[1][1]);
    assert_close!(-0.75, cov[0][1]);
    assert_close!(-0.75, cov[1][0]); // symmetric
  }

  #[test]
  fn test_vvv_covariance_known_values() {
    // Use points with known covariance: [0,0], [1,2], [2,4]
    // Perfect linear relationship Y = 2X
    let mut vvv = VVV::<f64>::new(2);
    vvv.add(&[0.0, 0.0]);
    vvv.add(&[1.0, 2.0]);
    vvv.add(&[2.0, 4.0]);

    // Mean: [1, 2]
    assert_close!(1.0, vvv.centroid[0]);
    assert_close!(2.0, vvv.centroid[1]);

    // SSD[0][0] = 1+0+1 = 2
    // SSD[1][1] = 4+0+4 = 8
    // SSD[0][1] = 1*2+0+1*2 = 4
    assert_close!(2.0, vvv.get_ssd(0, 0));
    assert_close!(8.0, vvv.get_ssd(1, 1));
    assert_close!(4.0, vvv.get_ssd(0, 1));

    let cov = vvv.covariance().unwrap();
    assert_close!(2.0 / 3.0, cov[0][0]);
    assert_close!(8.0 / 3.0, cov[1][1]);
    assert_close!(4.0 / 3.0, cov[0][1]);
    assert_close!(4.0 / 3.0, cov[1][0]);
  }

  #[test]
  fn test_vvv_add_cf_vvv_to_vvv() {
    let mut vvv1 = VVV::<f64>::new(2);
    vvv1.add(&[1.0, 2.0]);
    vvv1.add(&[3.0, 4.0]);

    let mut vvv2 = VVV::<f64>::new(2);
    vvv2.add(&[5.0, 6.0]);
    vvv2.add(&[7.0, 8.0]);

    // vvv1: centroid=[2,3], vvv2: centroid=[6,7]
    vvv1.add_cf(&vvv2);

    assert_eq!(vvv1.size, 4);
    // New centroid: [4, 5]
    assert_close!(4.0, vvv1.centroid[0]);
    assert_close!(5.0, vvv1.centroid[1]);

    // Verify total SSD matches VII
    let mut base = VII::<f64>::new(2);
    base.add(&[1.0, 2.0]);
    base.add(&[3.0, 4.0]);
    base.add(&[5.0, 6.0]);
    base.add(&[7.0, 8.0]);
    assert_close!(base.ssd(), vvv1.ssd());

    // Dim 0: [1,3,5,7], mean=4, SSD_00 = 9+1+1+9 = 20
    // Dim 1: [2,4,6,8], mean=5, SSD_11 = 9+1+1+9 = 20
    // Cross: SSD_01 = (-3)*(-3)+(-1)*(-1)+1*1+3*3 = 9+1+1+9 = 20
    assert_close!(20.0, vvv1.get_ssd(0, 0));
    assert_close!(20.0, vvv1.get_ssd(1, 1));
    assert_close!(20.0, vvv1.get_ssd(0, 1));
  }

  #[test]
  fn test_vvv_add_cf_empty() {
    let mut empty = VVV::<f64>::new(3);

    let mut vvv = VVV::<f64>::new(3);
    vvv.add(&[1.0, 2.0, 3.0]);
    vvv.add(&[4.0, 5.0, 6.0]);

    empty.add_cf(&vvv);

    assert_eq!(empty.size, 2);
    assert_close!(2.5, empty.centroid[0]);
    assert_close!(3.5, empty.centroid[1]);
    assert_close!(4.5, empty.centroid[2]);
    // SSD per dim: (1-2.5)²+(4-2.5)² = 2.25+2.25 = 4.5
    assert_close!(4.5, empty.get_ssd(0, 0));
    assert_close!(4.5, empty.get_ssd(1, 1));
    assert_close!(4.5, empty.get_ssd(2, 2));
    // Cross: all dims move together
    assert_close!(4.5, empty.get_ssd(0, 1));
    assert_close!(4.5, empty.get_ssd(0, 2));
    assert_close!(4.5, empty.get_ssd(1, 2));
  }

  #[test]
  fn test_vvv_add_cf_vvv_to_vii() {
    // VII.add_cf should work with VVV as the "other"
    let mut base = VII::<f64>::new(3);
    base.add(&[1.0, 2.0, 3.0]);
    base.add(&[3.0, 4.0, 5.0]);

    let mut vvv = VVV::<f64>::new(3);
    vvv.add(&[5.0, 6.0, 7.0]);
    vvv.add(&[7.0, 8.0, 9.0]);

    base.add_cf(&vvv);

    assert_eq!(base.size(), 4);
    assert_close!(4.0, base.centroid()[0]);
    assert_close!(5.0, base.centroid()[1]);
    assert_close!(6.0, base.centroid()[2]);

    let mut ref_cf = VII::<f64>::new(3);
    ref_cf.add(&[1.0, 2.0, 3.0]);
    ref_cf.add(&[3.0, 4.0, 5.0]);
    ref_cf.add(&[5.0, 6.0, 7.0]);
    ref_cf.add(&[7.0, 8.0, 9.0]);
    assert_close!(ref_cf.ssd(), base.ssd());
  }

  #[test]
  fn test_vvv_vs_vii_total_ssd() {
    // VVV total SSD (trace) should always match VII total SSD for the same data
    let points: Vec<Vec<f64>> = vec![
      vec![1.0, 2.0, 3.0],
      vec![3.0, 4.0, 5.0],
      vec![5.0, 6.0, 7.0],
      vec![2.0, 8.0, 1.0],
      vec![4.0, 0.0, 9.0],
    ];

    let mut vvv = VVV::<f64>::new(3);
    let mut base = VII::<f64>::new(3);
    for p in &points {
      vvv.add(p);
      base.add(p);
    }

    assert_close!(base.ssd(), vvv.ssd());
  }

  #[test]
  fn test_vvv_reset() {
    let mut vvv = VVV::<f64>::new(3);
    vvv.add(&[1.0, 2.0, 3.0]);
    vvv.add(&[4.0, 5.0, 6.0]);

    vvv.reset();

    assert_eq!(vvv.size, 0);
    assert_close!(0.0, vvv.centroid[0]);
    assert_close!(0.0, vvv.get_ssd(0, 0));
    assert_close!(0.0, vvv.get_ssd(0, 1));
    assert_close!(0.0, vvv.ssd());
  }

  #[test]
  fn test_vvv_variance_empty() {
    let vvv = VVV::<f64>::new(3);
    assert_close!(0.0, vvv.variance(0));
    assert_close!(0.0, vvv.variance(2));
  }

  #[test]
  fn test_vvv_variance_single_point() {
    let mut vvv = VVV::<f64>::new(3);
    vvv.add(&[1.0, 2.0, 3.0]);
    assert_close!(0.0, vvv.variance(0));
    assert_close!(0.0, vvv.variance(1));
    assert_close!(0.0, vvv.variance(2));
  }

  #[test]
  fn test_vvv_ssd_upper() {
    let mut vvv = VVV::<f64>::new(2);
    vvv.add(&[1.0, 2.0]);
    vvv.add(&[4.0, 1.0]);

    // ssd_upper() returns flat upper-triangular storage directly.
    // For dim=2: indices are (0,0)=0, (0,1)=1, (1,1)=2
    let mat = vvv.ssd_upper().unwrap();
    assert_eq!(mat.len(), 3); // dim*(dim+1)/2 = 2*3/2 = 3
    assert_close!(4.5, mat[0]); // (0,0)
    assert_close!(-1.5, mat[1]); // (0,1)
    assert_close!(0.5, mat[2]); // (1,1)
  }

  #[test]
  fn test_vvv_add_cf_preserves_cross_products() {
    // Key test: merging VVV with VVV preserves cross-product elements exactly
    let mut vvv1 = VVV::<f64>::new(2);
    // Points with negative correlation
    vvv1.add(&[0.0, 10.0]);
    vvv1.add(&[10.0, 0.0]);

    let mut vvv2 = VVV::<f64>::new(2);
    vvv2.add(&[0.0, 10.0]);
    vvv2.add(&[10.0, 0.0]);

    vvv1.add_cf(&vvv2);

    assert_eq!(vvv1.size, 4);
    // Centroid: [5, 5]
    assert_close!(5.0, vvv1.centroid[0]);
    assert_close!(5.0, vvv1.centroid[1]);

    // Dim 0: [0,10,0,10], mean=5, SSD_00 = 25+25+25+25 = 100
    // Dim 1: [10,0,10,0], mean=5, SSD_11 = 25+25+25+25 = 100
    // Cross: SSD_01 = (-5)*(5)+(5)*(-5)+(-5)*(5)+(5)*(-5) = -25-25-25-25 = -100
    assert_close!(100.0, vvv1.get_ssd(0, 0));
    assert_close!(100.0, vvv1.get_ssd(1, 1));
    assert_close!(-100.0, vvv1.get_ssd(0, 1));
    assert_close!(-100.0, vvv1.get_ssd(1, 0));
  }

  #[test]
  fn test_vvv_add_cf_asymmetric_data() {
    // Different variances and non-zero correlation
    let mut vvv1 = VVV::<f64>::new(2);
    vvv1.add(&[0.0, 0.0]);
    vvv1.add(&[2.0, 0.0]);

    let mut vvv2 = VVV::<f64>::new(2);
    vvv2.add(&[1.0, 1.0]);
    vvv2.add(&[3.0, 3.0]);

    vvv1.add_cf(&vvv2);

    assert_eq!(vvv1.size, 4);
    // Centroid: [1.5, 1.0]
    assert_close!(1.5, vvv1.centroid[0]);
    assert_close!(1.0, vvv1.centroid[1]);

    // Dim 0: [0,2,1,3], mean=1.5, SSD_00 = 2.25+0.25+0.25+2.25 = 5.0
    // Dim 1: [0,0,1,3], mean=1, SSD_11 = 1+1+0+4 = 6.0
    // Cross: SSD_01 = (-1.5)*(-1)+(0.5)*(-1)+(-0.5)*(0)+(1.5)*(2) = 1.5-0.5+0+3.0 = 4.0
    assert_close!(5.0, vvv1.get_ssd(0, 0));
    assert_close!(6.0, vvv1.get_ssd(1, 1));
    assert_close!(4.0, vvv1.get_ssd(0, 1));

    // Verify total SSD matches VII
    let mut base = VII::<f64>::new(2);
    base.add(&[0.0, 0.0]);
    base.add(&[2.0, 0.0]);
    base.add(&[1.0, 1.0]);
    base.add(&[3.0, 3.0]);
    assert_close!(base.ssd(), vvv1.ssd());
  }

  #[test]
  fn test_vvv_various_dimensions() {
    let dims = [1, 2, 5, 8, 16];
    for &dim in &dims {
      let mut vvv = VVV::<f64>::new(dim);
      for i in 0..10 {
        let point: Vec<f64> = (0..dim).map(|j| (i * dim + j) as f64).collect();
        vvv.add(&point);
      }
      assert_eq!(vvv.size, 10);
      // Total SSD should be non-negative
      assert!(vvv.ssd() >= 0.0);
      // Each diagonal SSD should be non-negative
      for i in 0..dim {
        assert!(vvv.get_ssd(i, i) >= 0.0, "ssd[{}][{}] < 0", i, i);
      }
      // Covariance matrix should be symmetric
      let cov = vvv.covariance().unwrap();
      for i in 0..dim {
        for j in 0..dim {
          assert_close!(
            cov[i][j],
            cov[j][i],
            "covariance not symmetric at [{},{}]",
            i,
            j
          );
        }
      }
    }
  }

  #[test]
  fn test_vvv_covariance_empty() {
    let vvv = VVV::<f64>::new(3);
    let cov = vvv.covariance().unwrap();
    assert_eq!(cov.len(), 3);
    for i in 0..3 {
      for j in 0..3 {
        assert_close!(0.0, cov[i][j]);
      }
    }
  }

  #[test]
  fn test_vvv_add_cf_vvv_to_vvi() {
    // VVI.add_cf should work with VVV as the "other"
    // VVI will fall back to diagonal from ssd_upper (no cross-product info from VVV)
    let mut vvi = VVI::<f64>::new(2);
    vvi.add(&[1.0, 2.0]);
    vvi.add(&[3.0, 4.0]);

    let mut vvv = VVV::<f64>::new(2);
    vvv.add(&[5.0, 6.0]);
    vvv.add(&[7.0, 8.0]);

    vvi.add_cf(&vvv);

    assert_eq!(vvi.size(), 4);
    assert_close!(4.0, vvi.centroid()[0]);
    assert_close!(5.0, vvi.centroid()[1]);

    // Total SSD should match VII
    let mut base = VII::<f64>::new(2);
    base.add(&[1.0, 2.0]);
    base.add(&[3.0, 4.0]);
    base.add(&[5.0, 6.0]);
    base.add(&[7.0, 8.0]);
    assert_close!(base.ssd(), vvi.ssd());
  }

  #[test]
  fn test_vvv_add_cf_vvi_to_vvv_preserves_diagonal_ssd() {
    // Regression test: VVV::add_cf reads other.ssd_upper() using upper-triangular
    // indexing, so VVI must provide the same layout.
    let mut vvi = VVI::<f64>::new(3);
    vvi.add(&[1.0, 2.0, 3.0]);
    vvi.add(&[4.0, 5.0, 6.0]);

    let mut vvv = VVV::<f64>::new(3);
    vvv.add_cf(&vvi);

    assert_eq!(vvv.size, 2);
    assert_close!(2.5, vvv.centroid[0]);
    assert_close!(3.5, vvv.centroid[1]);
    assert_close!(4.5, vvv.centroid[2]);
    assert_close!(4.5, vvv.get_ssd(0, 0));
    assert_close!(4.5, vvv.get_ssd(1, 1));
    assert_close!(4.5, vvv.get_ssd(2, 2));
    assert_close!(0.0, vvv.get_ssd(0, 1));
    assert_close!(0.0, vvv.get_ssd(0, 2));
    assert_close!(0.0, vvv.get_ssd(1, 2));
  }

  #[test]
  fn test_vvv_add_cf_vii_to_vvv_preserves_isotropic_diagonal_ssd() {
    // Regression test for the same layout mismatch through VII.
    let mut base = VII::<f64>::new(3);
    base.add(&[1.0, 2.0, 3.0]);
    base.add(&[4.0, 5.0, 6.0]);

    let expected_per_dim = base.ssd() / 3.0;

    let mut vvv = VVV::<f64>::new(3);
    vvv.add_cf(&base);

    assert_eq!(vvv.size, 2);
    assert_close!(2.5, vvv.centroid[0]);
    assert_close!(3.5, vvv.centroid[1]);
    assert_close!(4.5, vvv.centroid[2]);
    assert_close!(expected_per_dim, vvv.get_ssd(0, 0));
    assert_close!(expected_per_dim, vvv.get_ssd(1, 1));
    assert_close!(expected_per_dim, vvv.get_ssd(2, 2));
    assert_close!(0.0, vvv.get_ssd(0, 1));
    assert_close!(0.0, vvv.get_ssd(0, 2));
    assert_close!(0.0, vvv.get_ssd(1, 2));
  }

  #[test]
  fn test_vvv_add_cf_empty_other() {
    // Merging empty VVV into non-empty VVV should be a no-op
    let mut vvv = VVV::<f64>::new(2);
    vvv.add(&[1.0, 2.0]);
    vvv.add(&[4.0, 1.0]);

    let empty = VVV::<f64>::new(2);

    let saved_size = vvv.size;
    let saved_centroid = vvv.centroid.clone();
    let saved_ssd = vvv.ssd();

    vvv.add_cf(&empty);

    assert_eq!(vvv.size, saved_size);
    assert_close!(saved_centroid[0], vvv.centroid[0]);
    assert_close!(saved_ssd, vvv.ssd());
  }

  #[test]
  fn test_vvv_clone() {
    let mut vvv = VVV::<f64>::new(2);
    vvv.add(&[1.0, 2.0]);
    vvv.add(&[4.0, 1.0]);

    let vvv2 = vvv.clone();

    assert_eq!(vvv.size, vvv2.size);
    assert_close!(vvv.centroid[0], vvv2.centroid[0]);
    assert_close!(vvv.get_ssd(0, 0), vvv2.get_ssd(0, 0));
    assert_close!(vvv.get_ssd(0, 1), vvv2.get_ssd(0, 1));

    // Mutate original, clone should be unaffected
    vvv.add(&[5.0, 5.0]);
    assert_eq!(vvv2.size, 2);
  }

  #[test]
  fn test_vvv_upper_triangular_storage() {
    // Verify the internal flat storage layout and symmetric access
    let mut vvv = VVV::<f64>::new(3);
    vvv.add(&[1.0, 2.0, 3.0]);
    vvv.add(&[4.0, 5.0, 6.0]);

    // Flat storage: dim*(dim+1)/2 = 3*4/2 = 6 elements
    assert_eq!(vvv.ssd_upper.len(), 6);

    // Verify flat index formula: idx(i,j) = i*d - i*(i-1)/2 + (j-i)
    // (0,0)=0, (0,1)=1, (0,2)=2, (1,1)=3, (1,2)=4, (2,2)=5
    assert_close!(vvv.ssd_upper[0], vvv.get_ssd(0, 0));
    assert_close!(vvv.ssd_upper[1], vvv.get_ssd(0, 1));
    assert_close!(vvv.ssd_upper[2], vvv.get_ssd(0, 2));
    assert_close!(vvv.ssd_upper[3], vvv.get_ssd(1, 1));
    assert_close!(vvv.ssd_upper[4], vvv.get_ssd(1, 2));
    assert_close!(vvv.ssd_upper[5], vvv.get_ssd(2, 2));

    // Verify symmetric access
    assert_close!(vvv.get_ssd(0, 1), vvv.get_ssd(1, 0));
    assert_close!(vvv.get_ssd(0, 2), vvv.get_ssd(2, 0));
    assert_close!(vvv.get_ssd(1, 2), vvv.get_ssd(2, 1));
  }
}
