use crate::{
  cf_tree::CFTree,
  cluster_feature::{ClusterFeature, VII},
  distance::CFDistance,
  types::Float,
};

pub struct Betula<F: Float, CF: ClusterFeature<F>, D: CFDistance<F, CF>, A: CFDistance<F, CF>> {
  pub cf_tree: CFTree<F, CF, D, A>,
}
impl<F: Float, D: CFDistance<F, VII<F>>, A: CFDistance<F, VII<F>>> Betula<F, VII<F>, D, A> {
  #[must_use]
  pub fn new(
    data: &[&[F]],
    dist_fun: D,
    abs_fun: A,
    cap: usize,
    dim: usize,
    maxleaves: usize,
    threshold: F,
  ) -> Betula<F, VII<F>, D, A> {
    let mut cftree = CFTree::new(dist_fun, abs_fun, cap, dim, maxleaves, threshold);
    data.iter().for_each(|row| {
      assert!(row.len() >= dim);
      unsafe {
        cftree.insert_unchecked(row);
      }
    });
    Betula { cf_tree: cftree }
  }
}

#[cfg(test)]
mod tests {
  use crate::{betula::Betula, distance::CentroidEuclideanDistance};

  #[test]
  fn betula_constructs_tree_from_data() {
    let data: Vec<Vec<f64>> = vec![
      vec![1.0, 2.5],
      vec![2.0, 3.0],
      vec![3.0, 4.0],
      vec![3.0, 4.2],
      vec![1.0, 2.5],
    ];
    let data_refs: Vec<&[f64]> = data.iter().map(|v| v.as_slice()).collect::<Vec<_>>();
    let betula = Betula::new(
      &data_refs,
      CentroidEuclideanDistance::new(),
      CentroidEuclideanDistance::new(),
      10,
      2,
      10,
      0.0,
    );
    assert_eq!(betula.cf_tree.leaf_entries().len(), 4);
  }
}
