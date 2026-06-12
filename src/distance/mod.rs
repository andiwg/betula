use crate::{cluster_feature::ClusterFeature, types::Float};

mod centroid;
mod intercluster;
mod radius;
mod regression;
mod variance;
mod vector_distance;

pub use centroid::{CentroidEuclideanDistance, CentroidManhattanDistance};
pub use intercluster::{AverageInterclusterDistance, AverageIntraclusterDistance};
pub use radius::RadiusDistance;
pub use variance::VarianceIncreaseDistance;
pub use vector_distance::{Manhattan, SqEuclidean};

// ── Traits ──

/// Distance function for BIRCH / BETULA clustering.
///
/// All implementations use **squared distances** for performance.
///
/// Each method takes a dimensionality hint `d` to skip unused trailing
/// components when a `ClusterFeature` may have been allocated with a
/// larger dimensionality than the actual data.
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
pub trait CFDistance<F: Float, C: ClusterFeature<F>> {
  /// Distance of a point to a cluster feature (CF node).
  fn sq_dist(&self, cf1: &C, x: &[F], d: usize) -> F;
  /// Distance between two cluster features (used for internal-node merges).
  fn sq_dist_cf(&self, cf1: &C, cf2: &C, d: usize) -> F;
}

/// Distance between two raw vectors.
///
/// Implemented for 8-lane unrolled squared-Euclidean and Manhattan distances.
pub trait VectorDistance {
  /// Compute the distance between two vectors of length `d`.
  fn dist<F: Float>(vec1: &[F], vec2: &[F], d: usize) -> F;
}
