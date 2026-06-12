mod vii;
mod vvi;
mod vvv;

use crate::types::Float;

pub trait ClusterFeature<F: Float>
where
  Self: Clone,
{
  fn new(dim: usize) -> Self;
  fn size(&self) -> usize;
  fn centroid(&self) -> &[F];
  fn ssd(&self) -> F;
  /// Per-dimension SSDs.
  /// Returns `None` if the implementation doesn't store per-dimension SSDs inline.
  /// - VVI: returns `Some(&self.ssd)` (stored per-dimension, O(1))
  /// - VVV: returns `None` (stored as cross-products, needs extraction)
  /// - VII: returns `None` (stores scalar SSD only)
  fn ssd_per_dim(&self) -> Option<&[F]>;
  /// Upper-triangular SSD storage, including the diagonal, in flat row-major form.
  ///
  /// Layout contract: elements are stored row by row, each row `i` containing
  /// `dim - i` entries for `(i, i), (i, i+1), …, (i, dim-1)`.
  ///
  /// Flat index of `(i, j)` where `j >= i`:
  /// `idx = i * dim - i * (i - 1) / 2 + (j - i)`
  ///
  /// Total length: `dim * (dim + 1) / 2`.
  ///
  /// Returns `None` if the implementation doesn't store upper-triangular data inline.
  /// - VVV: returns `Some(&self.ssd_upper)` (raw storage, O(1))
  /// - VVI: returns `None` (stored as diagonal Vec, not upper-triangular layout)
  /// - VII: returns `None` (stores scalar SSD only)
  fn ssd_upper(&self) -> Option<&[F]>;
  fn variance(&self, d: usize) -> F;
  /// Returns the full covariance matrix, or `None` if this feature type
  /// does not store cross-product information.
  ///
  /// Only [`VVV`] stores cross-products and can return a covariance matrix.
  /// [`VII`] and [`VVI`] return `None`.
  ///
  /// [`VVV`]: crate::cluster_feature::VVV
  /// [`VII`]: crate::cluster_feature::VII
  /// [`VVI`]: crate::cluster_feature::VVI
  fn covariance(&self) -> Option<Vec<Vec<F>>>;
  fn reset(&mut self);
  fn add(&mut self, x: &[F]);
  fn add_cf<CF: ClusterFeature<F>>(&mut self, other: &CF);
}

pub use vii::VII;
pub use vvi::VVI;
pub use vvv::VVV;
