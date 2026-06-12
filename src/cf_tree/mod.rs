//! CF-Tree core: struct, insert operations, and helper modules.

mod insert;
mod rebuild;
mod regression;
mod split;

use crate::cluster_feature::ClusterFeature;
use crate::distance::CFDistance;
use crate::types::Float;
use std::marker::PhantomData;

/// Generic tree node holding a cluster feature and child references.
pub struct CFNode<F: Float, C: ClusterFeature<F>> {
  cf: C,
  children: Vec<usize>,
  is_leaf: bool,
  // Note: PhantomData<F> is required because F is only used in the trait bound
  // C: ClusterFeature<F>, not in a direct field. Cannot remove without breaking
  // compilation.
  _marker: PhantomData<F>,
}
impl<F: Float, C: ClusterFeature<F>> CFNode<F, C> {
  #[must_use]
  pub fn new(dim: usize, capacity: usize, is_leaf: bool) -> Self {
    Self {
      cf: C::new(dim),
      children: Vec::with_capacity(capacity),
      is_leaf,
      _marker: PhantomData,
    }
  }

  pub fn is_leaf(&self) -> bool {
    self.is_leaf
  }

  pub fn add_child(&mut self, child_id: usize) {
    self.children.push(child_id);
  }

  pub fn children(&self) -> &[usize] {
    &self.children
  }

  pub fn get_child(&self, i: usize) -> usize {
    *self.children.get(i).unwrap()
  }

  pub fn num_childs(&self) -> usize {
    self.children.len()
  }

  pub fn as_cf(&self) -> &C {
    &self.cf
  }

  pub fn as_mut_cf(&mut self) -> &mut C {
    &mut self.cf
  }
}

// ── CFTree struct ──

pub struct CFTree<F: Float, CF: ClusterFeature<F>, D: CFDistance<F, CF>, A: CFDistance<F, CF>> {
  dist_function: D,
  abs_function: A,
  capacity: usize,
  dim: usize,
  maxleaves: usize,
  root: usize,
  threshold: F,
  rebuild_count: usize,
  nodes: Vec<CFNode<F, CF>>,
  leaf_entries: Vec<CF>,
}

// ── Core impl: constructor + insert ──

impl<F: Float, CF: ClusterFeature<F>, D: CFDistance<F, CF>, A: CFDistance<F, CF>>
  CFTree<F, CF, D, A>
{
  #[must_use]
  pub fn new(dist: D, abs: A, capacity: usize, dim: usize, maxleaves: usize, threshold: F) -> Self {
    Self {
      dist_function: dist,
      abs_function: abs,
      capacity,
      dim,
      maxleaves,
      root: 0,
      threshold,
      rebuild_count: 0,
      nodes: vec![CFNode::new(dim, capacity, true)],
      leaf_entries: Vec::with_capacity(maxleaves),
    }
  }

  pub fn insert(&mut self, x: &[F]) {
    assert!(
      x.len() >= self.dim,
      "point dimension mismatch: expected at least {}, got {}",
      self.dim,
      x.len()
    );
    unsafe {
      self.insert_unchecked(x);
    }
  }

  pub fn dim(&self) -> usize {
    self.dim
  }

  pub fn root(&self) -> usize {
    self.root
  }

  pub fn nodes(&self) -> &[CFNode<F, CF>] {
    &self.nodes
  }

  pub fn leaf_entries(&self) -> &[CF] {
    &self.leaf_entries
  }

  pub fn rebuild_count(&self) -> usize {
    self.rebuild_count
  }

  // ── Point insertion (unsafe: caller must validate dimension) ──

  /// # Safety
  ///
  /// The caller must guarantee that `x.len() >= self.dim`.
  pub unsafe fn insert_unchecked(&mut self, x: &[F]) {
    let (mut path, leaf_idx) = insert::find_leaf_path(self, x);
    let overflow = insert::insert_into_leaf(self, leaf_idx, x);
    if let Some(cf) = overflow {
      insert::handle_overflow_point(self, cf, &mut path, x);
    }
    insert::update_path(self, &path, x);
    if self.leaf_entries.len() > self.maxleaves {
      self.rebuild_tree();
    }
  }

  // ── CF insertion (unsafe: caller must validate dimension) ──

  /// # Safety
  ///
  /// The caller must guarantee that `cf.centroid().len() >= self.dim`.
  pub unsafe fn insert_unchecked_cf(&mut self, cf: CF) {
    let (mut path, leaf_idx) = insert::find_leaf_path_cf(self, &cf);
    let overflow = insert::insert_into_leaf_cf(self, leaf_idx, &cf);
    if let Some(new_cf) = overflow {
      insert::handle_overflow_cf(self, new_cf, &mut path, &cf);
    }
    insert::update_path_cf(self, &path, &cf);
    if self.leaf_entries.len() > self.maxleaves {
      self.rebuild_tree();
    }
  }
}

// ── Tests ──

#[cfg(test)]
mod tests {
  use crate::cluster_feature::ClusterFeature;
  use crate::utils;
  use crate::{cf_tree::CFTree, distance::CentroidEuclideanDistance};

  fn create_data_det() -> Vec<Vec<f64>> {
    vec![
      vec![1.0, 2.5],
      vec![2.0, 3.0],
      vec![3.0, 4.0],
      vec![3.0, 4.2],
      vec![1.0, 2.5],
    ]
  }

  #[test]
  fn flat_tree_one_object_per_leaf() {
    let data = create_data_det();
    let mut tree = CFTree::new(
      CentroidEuclideanDistance::new(),
      CentroidEuclideanDistance::new(),
      10,
      2,
      10,
      0.,
    );
    for v in data.iter() {
      tree.insert(v);
    }
    assert_eq!(tree.leaf_entries.len(), 4);
    assert_eq!(tree.threshold, 0.);
    utils::verify_size(&data, tree.nodes.get(tree.root).unwrap().as_cf());
    println!(
      "{}",
      utils::calc_error_for_mean(&data, tree.nodes.get(tree.root).unwrap().as_cf())
    );
    utils::verify_node(&tree.nodes[tree.root], &tree);
  }

  #[test]
  #[should_panic(expected = "point dimension mismatch")]
  fn test_insert_rejects_dimension_mismatch() {
    let mut tree = CFTree::<
      f64,
      crate::cluster_feature::VII<f64>,
      CentroidEuclideanDistance<f64, crate::cluster_feature::VII<f64>>,
      CentroidEuclideanDistance<f64, crate::cluster_feature::VII<f64>>,
    >::new(
      CentroidEuclideanDistance::new(),
      CentroidEuclideanDistance::new(),
      10,
      3,
      10,
      0.0,
    );

    tree.insert(&[1.0, 2.0]);
  }

  #[test]
  fn test_basecf_first_insertion() {
    // Verify that add() works correctly on first insertion (size == 0).
    use crate::cluster_feature::VII;

    let dim = 2;
    let mut cf = VII::new(dim);
    assert_eq!(cf.size(), 0, "initial size should be 0");

    let x = vec![3.0, 4.0];
    cf.add(&x);
    assert_eq!(cf.size(), 1, "size should be 1 after first add");
    let expected_centroid: Vec<f64> = vec![3.0, 4.0];
    for i in 0..dim {
      let diff = (cf.centroid()[i] - expected_centroid[i]).abs();
      assert!(
        diff < 1e-10_f64,
        "centroid[{}] should be {}, got {}, diff={}",
        i,
        expected_centroid[i],
        cf.centroid()[i],
        diff
      );
    }
    let ssd_val: f64 = cf.ssd();
    assert!(
      ssd_val.abs() < 1e-10_f64,
      "SSD should be 0 after first add, got {}",
      ssd_val
    );

    // Verify that add_cf() also works correctly on first insertion
    let mut cf2 = VII::new(dim);
    let other = VII::new(dim);
    let mut other_copy = other.clone();
    other_copy.add(&[1.0, 2.0]);
    cf2.add_cf(&other_copy);
    assert_eq!(cf2.size(), 1, "size should be 1 after first add_cf");
  }
}
