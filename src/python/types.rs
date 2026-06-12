use pyo3::PyResult;
use pyo3::exceptions::PyValueError;

/// Cluster feature type selected from Python.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FeatureName {
  VII,
  VVI,
  VVV,
}

impl FeatureName {
  pub fn parse(name: &str) -> PyResult<Self> {
    match name.to_lowercase().as_str() {
      "vii" => Ok(Self::VII),
      "vvi" => Ok(Self::VVI),
      "vvv" => Ok(Self::VVV),
      _ => Err(PyValueError::new_err(format!(
        "Unknown feature '{}'. Valid options: vii, vvi, vvv",
        name
      ))),
    }
  }
}

/// Distance measure name used in Python API.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DistanceName {
  Euclidean,
  Manhattan,
  AvgIntercluster,
  AvgIntracluster,
  VarianceIncrease,
  Radius,
}

impl DistanceName {
  pub fn parse(name: &str) -> PyResult<Self> {
    match name.to_lowercase().as_str() {
      "euclidean" => Ok(Self::Euclidean),
      "manhattan" => Ok(Self::Manhattan),
      "avgintercluster" => Ok(Self::AvgIntercluster),
      "avgintracluster" => Ok(Self::AvgIntracluster),
      "varianceincrease" => Ok(Self::VarianceIncrease),
      "radius" => Ok(Self::Radius),
      _ => Err(PyValueError::new_err(format!(
        "Unknown distance '{}'. Valid options: euclidean, manhattan, avgintercluster, avgintracluster, varianceincrease, radius",
        name
      ))),
    }
  }
}
