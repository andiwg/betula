//! # BETULA — Fast, streamable data clustering
//!
//! Rust implementation of BETULA, an improved version of the BIRCH algorithm
//! with numerically stable cluster features. Fast and streamable data aggregation
//! and clustering.
//!
//! ## What is BETULA?
//!
//! BETULA introduces improved [Clustering Feature (CF) trees](https://en.wikipedia.org/wiki/CF_tree)
//! that avoid the catastrophic cancellation present in the original BIRCH algorithm.
//! The key contribution is a replacement cluster feature that is numerically stable,
//! not much more expensive to maintain, and makes many computations simpler.
//!
//! **Reference:** Lang & Schubert, *"BETULA: Fast clustering of large data with
//! improved BIRCH CF-Trees"*, Information Systems 108 (2022),
//! [DOI: 10.1016/J.IS.2021.101918](https://doi.org/10.1016/J.IS.2021.101918)
//!
//! ## Quick Start
//!
//! ```
//! use betula::cf_tree::CFTree;
//! use betula::cluster_feature::{ClusterFeature, VII};
//! use betula::distance::CentroidEuclideanDistance;
//!
//! let mut tree: CFTree<f64, VII<f64>, _, _> = CFTree::new(
//!     CentroidEuclideanDistance::new(),
//!     CentroidEuclideanDistance::new(),
//!     32,   // capacity
//!     2,    // dimensions
//!     1000, // max leaves
//!     0.0,  // threshold
//! );
//!
//! for point in &[vec![1.0, 2.0], vec![1.1, 2.1], vec![10.0, 20.0]] {
//!     tree.insert(point);
//! }
//!
//! for (id, cf) in tree.leaf_entries().iter().enumerate() {
//!     println!("Cluster {}: {} points", id, cf.size());
//! }
//! ```
//!
//! ## Cluster Features
//!
//! Three feature types are available, each storing different levels of summary statistics:
//!
//! | Feature | Stores | Use case |
//! |---------|--------|----------|
//! | [`VII`] | Scalar SSD + centroid | Simple clustering, most common |
//! | [`VVI`] | Per-dimension SSD + centroid | Per-dimension variance matters |
//! | [`VVV`] | Full cross-product matrix + centroid | Covariance needed |
//!
//! ## Distance Measures
//!
//! Six distance functions implement the [`CFDistance`] trait:
//!
//! - [`CentroidEuclideanDistance`] — Squared Euclidean between centroids
//! - [`CentroidManhattanDistance`] — Manhattan distance between centroids
//! - [`AverageInterclusterDistance`] — Average squared distance to cluster points
//! - [`AverageIntraclusterDistance`] — Average pairwise distance within clusters
//! - [`VarianceIncreaseDistance`] — SSD increase from adding/merging
//! - [`RadiusDistance`] — Average radius of the resulting cluster
//!
//! ## Feature Flags
//!
//! | Flag | Description |
//! |------|-------------|
//! | `bench` | Enable the `bench` binary (macro benchmarks) |
//! | `python` | Enable Python bindings (requires `pyo3`) |

pub mod betula;
pub mod cf_tree;
pub mod cluster_feature;
pub mod distance;
pub mod types;
pub mod utils;

#[cfg(feature = "python")]
mod python;
