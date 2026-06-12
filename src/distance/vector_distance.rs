use crate::types::Float;

use super::VectorDistance;

// ── Vector distance implementations ──

/// Squared Euclidean distance between two vectors.
///
/// Uses 8-wide lane accumulation with FMA for numerical stability.
/// Falls back to scalar for dimensions not divisible by 8.
pub struct SqEuclidean;
impl VectorDistance for SqEuclidean {
  #[inline(always)]
  fn dist<F: Float>(a: &[F], b: &[F], d: usize) -> F {
    const LANES: usize = 8;
    let sd = d & !(LANES - 1);
    let mut vsum = [F::zero(); LANES];
    for i in (0..sd).step_by(LANES) {
      let (vv, cc) = (&a[i..(i + LANES)], &b[i..(i + LANES)]);
      for j in 0..LANES {
        unsafe {
          let (a, b) = (*vv.get_unchecked(j), *cc.get_unchecked(j));
          let diff = a - b;
          *vsum.get_unchecked_mut(j) = diff.mul_add(diff, *vsum.get_unchecked_mut(j)); // FMA
        }
      }
    }
    let mut sum = vsum.iter().copied().sum::<F>();
    if d > sd {
      sum += (sd..d)
        .map(|i| unsafe {
          let diff = *a.get_unchecked(i) - *b.get_unchecked(i);
          diff * diff
        })
        .sum()
    }
    sum
  }
}

/// Manhattan (L1) distance between two vectors.
///
/// Uses 8-wide lane accumulation. Falls back to scalar for dimensions
/// not divisible by 8.
pub struct Manhattan;
impl VectorDistance for Manhattan {
  #[inline(always)]
  fn dist<F: Float>(a: &[F], b: &[F], d: usize) -> F {
    const LANES: usize = 8;
    let sd = d & !(LANES - 1);
    let mut vsum = [F::zero(); LANES];
    for i in (0..sd).step_by(LANES) {
      let (vv, cc) = (&a[i..(i + LANES)], &b[i..(i + LANES)]);
      for j in 0..LANES {
        unsafe {
          let (a, b) = (*vv.get_unchecked(j), *cc.get_unchecked(j));
          let diff = (a - b).abs();
          *vsum.get_unchecked_mut(j) += diff;
        }
      }
    }
    let mut sum = vsum.iter().copied().sum::<F>();
    if d > sd {
      sum += (sd..d)
        .map(|i| unsafe { (*a.get_unchecked(i) - *b.get_unchecked(i)).abs() })
        .sum()
    }
    sum
  }
}

// ── Tests ──

#[cfg(test)]
mod tests {
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

  /// Reference implementation for correctness verification
  fn reference_sq_euclidean<F: Float>(a: &[F], b: &[F], d: usize) -> F {
    let mut sum = F::zero();
    for i in 0..d {
      let diff = a[i] - b[i];
      sum += diff * diff;
    }
    sum
  }

  /// Reference implementation for Manhattan distance
  fn reference_manhattan<F: Float>(a: &[F], b: &[F], d: usize) -> F {
    let mut sum = F::zero();
    for i in 0..d {
      let diff = a[i] - b[i];
      sum += diff.abs();
    }
    sum
  }

  /// Test a VectorDistance against a reference across various dimensions.
  fn test_vector_distance<D: VectorDistance, R>(
    name: &str,
    reference: R,
    dims: &[usize],
    a: &[f64],
    b: &[f64],
  ) where
    R: Fn(&[f64], &[f64], usize) -> f64,
  {
    for &dim in dims {
      let expected = reference(a, b, dim);
      let actual = D::dist(a, b, dim);
      assert_close!(
        expected,
        actual,
        "{} dim={}: expected={}, actual={}",
        name,
        dim,
        expected,
        actual
      );
    }
  }

  #[test]
  fn test_sq_euclidean() {
    // Exercises: dim 0 (empty), 1-7 (scalar fallback), 8 (one SIMD block),
    // 9 (block+scalar), 15, 16 (two blocks), 17, 64, 65
    let dims = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 64, 65];
    let a: Vec<f64> = (0..65).map(|i| (i as f64) * 0.1).collect();
    let b: Vec<f64> = (0..65).map(|i| ((i as f64) * 0.1).sin().abs()).collect();
    test_vector_distance::<SqEuclidean, _>("sq_euclidean", reference_sq_euclidean, &dims, &a, &b);
  }

  #[test]
  fn test_manhattan() {
    let dims = [0, 1, 2, 3, 4, 5, 8, 9, 15, 16, 17, 64, 65];
    let a: Vec<f64> = (0..65).map(|i| (i as f64) * 0.1).collect();
    let b: Vec<f64> = (0..65).map(|i| ((i as f64) * 0.1).sin().abs()).collect();
    test_vector_distance::<Manhattan, _>("manhattan", reference_manhattan, &dims, &a, &b);
  }

  #[test]
  fn test_manhattan_negative_values() {
    // |-1 - 1| + |2 - (-2)| = 2 + 4 = 6
    assert_close!(6.0_f64, Manhattan::dist(&[-1.0, 2.0], &[1.0, -2.0], 2));
  }
}
