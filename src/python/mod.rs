//! Python bindings for BETULA via pyo3.

use pyo3::prelude::*;

mod betula;
mod cf_tree;
mod cluster_feature;
mod types;

use betula::PyBetula;
use cf_tree::PyCFTree;
use cluster_feature::PyClusterFeature;

#[pymodule]
fn _betulars(m: &Bound<'_, PyModule>) -> PyResult<()> {
  m.add_class::<PyBetula>()?;
  m.add_class::<PyCFTree>()?;
  m.add_class::<PyClusterFeature>()?;
  m.add("__version__", env!("CARGO_PKG_VERSION"))?;
  Ok(())
}
