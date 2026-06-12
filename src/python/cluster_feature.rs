use pyo3::prelude::*;

use crate::cluster_feature::ClusterFeature;

/// Python-accessible cluster feature (read-only view).
#[pyclass(name = "ClusterFeature", module = "betulars", skip_from_py_object)]
#[derive(Clone)]
pub struct PyClusterFeature {
  pub size: usize,
  centroid: Vec<f64>,
  pub ssd: f64,
  ssd_per_dim: Vec<f64>,
  covariance: Option<Vec<Vec<f64>>>,
}

#[pymethods]
impl PyClusterFeature {
  #[getter]
  fn size(&self) -> usize {
    self.size
  }

  /// Cluster centroid as a numpy 1D array.
  #[getter]
  fn centroid<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<f64>> {
    numpy::PyArray1::from_slice(py, &self.centroid)
  }

  #[getter]
  fn ssd(&self) -> f64 {
    self.ssd
  }

  /// Per-dimension SSDs as a numpy 1D array.
  #[getter]
  fn ssd_per_dim<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<f64>> {
    numpy::PyArray1::from_slice(py, &self.ssd_per_dim)
  }

  /// Full covariance matrix as a numpy 2D array, or None if unavailable.
  ///
  /// Only VVV stores cross-product information and can return a covariance
  /// matrix. VII and VVI return None.
  #[getter]
  fn covariance<'py>(&self, py: Python<'py>) -> Option<Bound<'py, numpy::PyArray2<f64>>> {
    self
      .covariance
      .as_ref()
      .map(|mat| numpy::PyArray2::from_vec2(py, mat).unwrap())
  }

  /// Variance (ssd / size, or 0.0 if empty).
  #[getter]
  fn variance(&self) -> f64 {
    if self.size > 0 {
      self.ssd / self.size as f64
    } else {
      0.0
    }
  }

  fn __repr__(&self) -> String {
    format!(
      "ClusterFeature(size={}, centroid={:?}, ssd={:.4})",
      self.size, self.centroid, self.ssd
    )
  }
}

impl PyClusterFeature {
  /// Build from any ClusterFeature implementation.
  pub fn from_cf<CF: ClusterFeature<f64>>(cf: &CF) -> Self {
    Self {
      size: cf.size(),
      centroid: cf.centroid().to_vec(),
      ssd: cf.ssd(),
      ssd_per_dim: cf.ssd_per_dim().map(|s| s.to_vec()).unwrap_or_default(),
      covariance: cf.covariance(),
    }
  }
}
