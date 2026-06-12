use pyo3::prelude::*;

use crate::cf_tree::CFTree;
use crate::cluster_feature::{ClusterFeature, VII, VVI, VVV};
use crate::distance::{CFDistance, CentroidEuclideanDistance};

use super::cluster_feature::PyClusterFeature;
use super::types::FeatureName;

/// Extract contiguous f64 data from a 1D numpy array or sequence.
fn extract_point(point: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
  // Try numpy array first
  if let Ok(arr) = point.extract::<numpy::PyReadonlyArray1<f64>>() {
    return Ok(arr.as_slice()?.to_vec());
  }
  // Fall back to sequence protocol (list, tuple)
  point.extract()
}

/// Build a tree from flat data and extract results.
pub fn build_tree_and_extract<CF, D, A>(
  flat: &[f64],
  dim: usize,
  capacity: usize,
  maxleaves: usize,
  threshold: f64,
  dist: D,
  abs: A,
) -> (Vec<PyClusterFeature>, usize, f64)
where
  CF: ClusterFeature<f64>,
  D: CFDistance<f64, CF>,
  A: CFDistance<f64, CF>,
{
  let mut tree = CFTree::<f64, CF, D, A>::new(dist, abs, capacity, dim, maxleaves, threshold);

  let mut offset = 0;
  while offset + dim <= flat.len() {
    let row = &flat[offset..offset + dim];
    unsafe {
      tree.insert_unchecked(row);
    }
    offset += dim;
  }

  let leaf_entries = tree.leaf_entries();
  let clusters: Vec<PyClusterFeature> =
    leaf_entries.iter().map(PyClusterFeature::from_cf).collect();

  let total_points: usize = clusters.iter().map(|c| c.size).sum();
  let total_ssd: f64 = clusters.iter().map(|c| c.ssd).sum();
  let overall_variance = if total_points > 0 {
    total_ssd / total_points as f64
  } else {
    0.0
  };

  (clusters, tree.rebuild_count(), overall_variance)
}

/// Internal enum wrapping the three possible CF-tree specializations.
enum PyCFTreeInner {
  VII(
    CFTree<
      f64,
      VII<f64>,
      CentroidEuclideanDistance<f64, VII<f64>>,
      CentroidEuclideanDistance<f64, VII<f64>>,
    >,
  ),
  VVI(
    CFTree<
      f64,
      VVI<f64>,
      CentroidEuclideanDistance<f64, VVI<f64>>,
      CentroidEuclideanDistance<f64, VVI<f64>>,
    >,
  ),
  VVV(
    CFTree<
      f64,
      VVV<f64>,
      CentroidEuclideanDistance<f64, VVV<f64>>,
      CentroidEuclideanDistance<f64, VVV<f64>>,
    >,
  ),
}

impl PyCFTreeInner {
  fn new(
    dim: usize,
    capacity: usize,
    maxleaves: usize,
    threshold: f64,
    feature: FeatureName,
  ) -> Self {
    match feature {
      FeatureName::VII => Self::VII(CFTree::new(
        CentroidEuclideanDistance::new(),
        CentroidEuclideanDistance::new(),
        capacity,
        dim,
        maxleaves,
        threshold,
      )),
      FeatureName::VVI => Self::VVI(CFTree::new(
        CentroidEuclideanDistance::new(),
        CentroidEuclideanDistance::new(),
        capacity,
        dim,
        maxleaves,
        threshold,
      )),
      FeatureName::VVV => Self::VVV(CFTree::new(
        CentroidEuclideanDistance::new(),
        CentroidEuclideanDistance::new(),
        capacity,
        dim,
        maxleaves,
        threshold,
      )),
    }
  }

  fn insert(&mut self, point: &[f64]) {
    match self {
      Self::VII(t) => t.insert(point),
      Self::VVI(t) => t.insert(point),
      Self::VVV(t) => t.insert(point),
    }
  }

  fn leaf_clusters(&self) -> Vec<PyClusterFeature> {
    match self {
      Self::VII(t) => t
        .leaf_entries()
        .iter()
        .map(PyClusterFeature::from_cf)
        .collect(),
      Self::VVI(t) => t
        .leaf_entries()
        .iter()
        .map(PyClusterFeature::from_cf)
        .collect(),
      Self::VVV(t) => t
        .leaf_entries()
        .iter()
        .map(PyClusterFeature::from_cf)
        .collect(),
    }
  }

  fn num_clusters(&self) -> usize {
    match self {
      Self::VII(t) => t.leaf_entries().len(),
      Self::VVI(t) => t.leaf_entries().len(),
      Self::VVV(t) => t.leaf_entries().len(),
    }
  }

  fn rebuild_count(&self) -> usize {
    match self {
      Self::VII(t) => t.rebuild_count(),
      Self::VVI(t) => t.rebuild_count(),
      Self::VVV(t) => t.rebuild_count(),
    }
  }

  fn dim(&self) -> usize {
    match self {
      Self::VII(t) => t.dim(),
      Self::VVI(t) => t.dim(),
      Self::VVV(t) => t.dim(),
    }
  }
}

/// A CF-Tree for incremental clustering.
///
/// Insert points one at a time and access the resulting leaf clusters.
#[pyclass(name = "CFTree", module = "betulars")]
pub struct PyCFTree {
  inner: PyCFTreeInner,
}

#[pymethods]
impl PyCFTree {
  #[new]
  #[pyo3(signature = (dim, capacity = 32, maxleaves = 1000, threshold = 0.0, feature = "vii"))]
  fn new(
    dim: usize,
    capacity: usize,
    maxleaves: usize,
    threshold: f64,
    feature: &str,
  ) -> PyResult<Self> {
    Ok(Self {
      inner: PyCFTreeInner::new(
        dim,
        capacity,
        maxleaves,
        threshold,
        FeatureName::parse(feature)?,
      ),
    })
  }

  /// Insert a single data point into the tree.
  ///
  /// Args:
  ///     point: 1D numpy array or sequence of floats. Length must be >= the tree's dimensionality.
  fn insert(&mut self, point: &Bound<'_, PyAny>) -> PyResult<()> {
    let pt = extract_point(point)?;
    self.inner.insert(&pt);
    Ok(())
  }

  /// Leaf cluster entries as a list of ClusterFeature objects.
  #[getter]
  fn leaf_clusters(&self) -> Vec<PyClusterFeature> {
    self.inner.leaf_clusters()
  }

  /// Number of leaf clusters.
  #[getter]
  fn num_clusters(&self) -> usize {
    self.inner.num_clusters()
  }

  /// Number of times the tree has been rebuilt.
  #[getter]
  fn rebuild_count(&self) -> usize {
    self.inner.rebuild_count()
  }

  /// Dimensionality of the tree.
  #[getter]
  fn dim(&self) -> usize {
    self.inner.dim()
  }

  fn __repr__(&self) -> String {
    format!(
      "CFTree(dim={}, clusters={}, rebuilds={})",
      self.inner.dim(),
      self.inner.num_clusters(),
      self.inner.rebuild_count(),
    )
  }
}
