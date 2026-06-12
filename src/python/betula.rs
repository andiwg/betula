use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::cluster_feature::{VII, VVI, VVV};
use crate::distance::{
  AverageInterclusterDistance, AverageIntraclusterDistance, CentroidEuclideanDistance,
  CentroidManhattanDistance, RadiusDistance, VarianceIncreaseDistance,
};
use numpy::PyArrayMethods;

use super::cf_tree::build_tree_and_extract;
use super::cluster_feature::PyClusterFeature;
use super::types::{DistanceName, FeatureName};

/// Extract flat f64 data from a 2D numpy array (row-major / C-contiguous).
fn extract_array2(data: &Bound<'_, PyAny>) -> PyResult<(usize, usize, Vec<f64>)> {
  let arr: numpy::PyReadonlyArray2<f64> = data.extract()?;
  let dims = arr.dims();
  let rows = dims[0] as usize;
  let cols = dims[1] as usize;
  let flat = arr.to_vec()?;
  Ok((rows, cols, flat))
}

/// High-level BETULA clustering wrapper.
#[pyclass(name = "Betula", module = "betulars")]
pub struct PyBetula {
  clusters: Vec<PyClusterFeature>,
  num_rebuilds: usize,
  overall_variance: f64,
}

#[pymethods]
impl PyBetula {
  /// Construct a Betula model from data.
  ///
  /// Args:
  ///     data: 2D numpy array of shape (n_points, n_dims), dtype float64.
  ///     capacity: Node capacity (default 32).
  ///     maxleaves: Maximum leaf entries before rebuild (default 1000).
  ///     threshold: Leaf entry size threshold (default 0.0).
  ///     distance: Distance measure for tree routing (default "euclidean").
  ///     absorption: Distance measure for leaf absorption (default "euclidean").
  ///     feature: Cluster feature type — "vii", "vvi", or "vvv" (default "vii").
  #[new]
  #[pyo3(signature = (data, capacity = 32, maxleaves = 1000, threshold = 0.0, distance = "euclidean", absorption = "euclidean", feature = "vii"))]
  fn new(
    data: &Bound<'_, PyAny>,
    capacity: usize,
    maxleaves: usize,
    threshold: f64,
    distance: &str,
    absorption: &str,
    feature: &str,
  ) -> PyResult<Self> {
    let (_rows, dim, flat) = extract_array2(data)?;
    if flat.is_empty() {
      return Err(PyValueError::new_err("data must not be empty"));
    }
    if dim == 0 {
      return Err(PyValueError::new_err(
        "points must have at least 1 dimension",
      ));
    }

    let dist_kind = DistanceName::parse(distance)?;
    let abs_kind = DistanceName::parse(absorption)?;
    let feat_kind = FeatureName::parse(feature)?;

    macro_rules! build {
      ($CF:ty, $D:ty, $A:ty) => {
        build_tree_and_extract::<$CF, $D, $A>(
          &flat,
          dim,
          capacity,
          maxleaves,
          threshold,
          <$D>::new(),
          <$A>::new(),
        )
      };
    }

    macro_rules! dispatch_abs {
      ($CF:ty, $D:ty, $abs_kind:expr) => {
        match $abs_kind {
          DistanceName::Euclidean => build!($CF, $D, CentroidEuclideanDistance<f64, $CF>),
          DistanceName::Manhattan => build!($CF, $D, CentroidManhattanDistance<f64, $CF>),
          DistanceName::AvgIntercluster => build!($CF, $D, AverageInterclusterDistance<f64, $CF>),
          DistanceName::AvgIntracluster => build!($CF, $D, AverageIntraclusterDistance<f64, $CF>),
          DistanceName::VarianceIncrease => build!($CF, $D, VarianceIncreaseDistance<f64, $CF>),
          DistanceName::Radius => build!($CF, $D, RadiusDistance<f64, $CF>),
        }
      };
    }

    macro_rules! dispatch_dist {
      ($CF:ty, $dist_kind:expr, $abs_kind:expr) => {
        match $dist_kind {
          DistanceName::Euclidean => dispatch_abs!($CF, CentroidEuclideanDistance<f64, $CF>, $abs_kind),
          DistanceName::Manhattan => dispatch_abs!($CF, CentroidManhattanDistance<f64, $CF>, $abs_kind),
          DistanceName::AvgIntercluster => dispatch_abs!($CF, AverageInterclusterDistance<f64, $CF>, $abs_kind),
          DistanceName::AvgIntracluster => dispatch_abs!($CF, AverageIntraclusterDistance<f64, $CF>, $abs_kind),
          DistanceName::VarianceIncrease => dispatch_abs!($CF, VarianceIncreaseDistance<f64, $CF>, $abs_kind),
          DistanceName::Radius => dispatch_abs!($CF, RadiusDistance<f64, $CF>, $abs_kind),
        }
      };
    }

    let (clusters, num_rebuilds, overall_variance) = match feat_kind {
      FeatureName::VII => dispatch_dist!(VII<f64>, dist_kind, abs_kind),
      FeatureName::VVI => dispatch_dist!(VVI<f64>, dist_kind, abs_kind),
      FeatureName::VVV => dispatch_dist!(VVV<f64>, dist_kind, abs_kind),
    };

    Ok(Self {
      clusters,
      num_rebuilds,
      overall_variance,
    })
  }

  #[getter]
  fn leaf_clusters(&self) -> Vec<PyClusterFeature> {
    self.clusters.clone()
  }

  #[getter]
  fn num_clusters(&self) -> usize {
    self.clusters.len()
  }

  #[getter]
  fn rebuild_count(&self) -> usize {
    self.num_rebuilds
  }

  #[getter]
  fn overall_variance(&self) -> f64 {
    self.overall_variance
  }

  fn __repr__(&self) -> String {
    format!(
      "Betula(clusters={}, rebuilds={}, variance={:.4})",
      self.clusters.len(),
      self.num_rebuilds,
      self.overall_variance,
    )
  }
}
