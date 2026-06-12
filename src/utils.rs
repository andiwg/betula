use crate::{
  cf_tree::{CFNode, CFTree},
  cluster_feature::{ClusterFeature, VII},
  distance::{CentroidEuclideanDistance, SqEuclidean, VectorDistance},
};

pub fn calc_error_for_mean<CF: ClusterFeature<f64>>(ds: &[Vec<f64>], cf: &CF) -> f64 {
  let mut meansum = vec![0.; ds[0].len()];
  ds.iter()
    .for_each(|x| meansum.iter_mut().zip(x).for_each(|(c, x)| *c += *x));
  let center: Vec<_> = meansum.iter().map(|f| f / ds.len() as f64).collect();
  SqEuclidean::dist(&center, cf.centroid(), center.len())
}

pub fn verify_size(ds: &[Vec<f64>], cf: &VII<f64>) {
  assert_eq!(ds.len(), cf.size());
}

pub fn verify_node<CF: ClusterFeature<f64>>(
  node: &CFNode<f64, CF>,
  tree: &CFTree<f64, CF, CentroidEuclideanDistance<f64, CF>, CentroidEuclideanDistance<f64, CF>>,
) {
  if node.is_leaf() {
    // For leaf nodes, verify child entries directly
    let mut reconstructed_cf = CF::new(tree.dim());
    for entry_id in node.children() {
      let entry_cf = &tree.leaf_entries()[*entry_id];
      reconstructed_cf.add_cf(entry_cf);
    }
    let diff: f64 = node
      .as_cf()
      .centroid()
      .iter()
      .zip(reconstructed_cf.centroid())
      .map(|(a, b)| (a - b).abs())
      .sum();
    assert!(diff < 0.001, "Centroid mismatch in leaf node");
    assert!(
      (reconstructed_cf.ssd() - node.as_cf().ssd()).abs() < 0.001,
      "SSD mismatch in leaf node"
    );
  } else {
    // Reconstruct cluster feature from child nodes
    let mut reconstructed_cf = CF::new(tree.dim());
    for child_id in node.children() {
      let child_cf = tree.nodes()[*child_id].as_cf();
      reconstructed_cf.add_cf(child_cf);
    }
    let diff: f64 = node
      .as_cf()
      .centroid()
      .iter()
      .zip(reconstructed_cf.centroid())
      .map(|(a, b)| (a - b).abs())
      .sum();
    assert!(diff < 0.001, "Centroid mismatch in node");
    assert!(
      (reconstructed_cf.ssd() - node.as_cf().ssd()).abs() < 0.001,
      "SSD mismatch in node"
    );

    // Recursively verify all child nodes
    for child_id in node.children() {
      verify_node(&tree.nodes()[*child_id], tree);
    }
  }
}
