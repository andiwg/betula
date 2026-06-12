//! Regression tests for CFTree with golden values.

#[cfg(test)]
mod tests {
  use rand::rand_core::SeedableRng;
  use rand::{Rng, rngs::StdRng};

  use crate::cf_tree::CFTree;
  use crate::cluster_feature::{ClusterFeature, VII};
  use crate::distance::{
    AverageInterclusterDistance, AverageIntraclusterDistance, CFDistance,
    CentroidEuclideanDistance, CentroidManhattanDistance,
  };

  fn create_data(size: usize, dim: usize) -> Vec<Vec<f64>> {
    let mut data = Vec::with_capacity(size);
    let mut rng = StdRng::seed_from_u64(0xBAD5EEDu64).random_iter();
    for _ in 0..size {
      data.push(rng.by_ref().take(dim).collect::<Vec<f64>>());
    }
    data
  }

  /// Compute overall variance (sum of ssd / total points) from leaf entries.
  fn overall_variance<D: CFDistance<f64, VII<f64>>, A: CFDistance<f64, VII<f64>>>(
    tree: &CFTree<f64, VII<f64>, D, A>,
  ) -> f64 {
    let total_ssd: f64 = tree.leaf_entries().iter().map(|cf| cf.ssd()).sum();
    let total_points: usize = tree.leaf_entries().iter().map(|cf| cf.size()).sum();
    if total_points == 0 {
      0.0
    } else {
      total_ssd / total_points as f64
    }
  }

  /// Sorted cluster stats: (size, centroid, ssd) for each leaf.
  fn sorted_cluster_stats<D: CFDistance<f64, VII<f64>>, A: CFDistance<f64, VII<f64>>>(
    tree: &CFTree<f64, VII<f64>, D, A>,
  ) -> Vec<(usize, Vec<f64>, f64)> {
    let mut stats: Vec<_> = tree
      .leaf_entries()
      .iter()
      .map(|cf| (cf.size(), cf.centroid().to_vec(), cf.ssd()))
      .collect();
    stats.sort_by(|a, b| {
      a.1
        .iter()
        .zip(&b.1)
        .map(|(x, y)| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
        .find(|&o| o != std::cmp::Ordering::Equal)
        .unwrap_or(std::cmp::Ordering::Equal)
    });
    stats
  }

  /// Build a tree from data and assert it matches golden values.
  fn assert_regression<D: CFDistance<f64, VII<f64>> + Clone>(
    dist: D,
    n_points: usize,
    dim: usize,
    capacity: usize,
    maxleaves: usize,
    expected_leaves: usize,
    expected_variance: f64,
    expected_stats: &[(usize, &[f64], f64)],
  ) {
    let data = create_data(n_points, dim);
    let mut tree = CFTree::new(dist.clone(), dist, capacity, dim, maxleaves, 0.0);
    for v in data.iter() {
      tree.insert(v);
    }

    assert_eq!(
      tree.leaf_entries().len(),
      expected_leaves,
      "leaf count mismatch"
    );
    let ov = overall_variance(&tree);
    assert!(
      (ov - expected_variance).abs() < 1e-6,
      "overall variance mismatch: got {}",
      ov
    );

    let stats = sorted_cluster_stats(&tree);
    assert_eq!(stats.len(), expected_stats.len());
    for (i, ((sz, cent, ssd), &(esz, ecent, essd))) in stats.iter().zip(expected_stats).enumerate()
    {
      assert_eq!(*sz, esz, "cluster {} size mismatch", i);
      for (d, (c, ec)) in cent.iter().zip(ecent.iter()).enumerate() {
        assert!(
          (c - ec).abs() < 1e-4,
          "cluster {} centroid[{}] mismatch: got {} expected {}",
          i,
          d,
          c,
          ec
        );
      }
      assert!(
        (ssd - essd).abs() < 1e-4,
        "cluster {} ssd mismatch: got {} expected {}",
        i,
        ssd,
        essd
      );
    }
  }

  // ── Regression tests ──

  #[test]
  fn regression_euclidean_small() {
    assert_regression(
      CentroidEuclideanDistance::new(),
      50,
      2,
      10,
      10, // n_points, dim, capacity, maxleaves
      7,
      0.0230103956, // expected_leaves, expected_variance
      &[
        (12, &[0.262528, 0.266949], 0.386061),
        (13, &[0.269360, 0.659733], 0.347445),
        (7, &[0.368375, 0.921907], 0.046654),
        (7, &[0.675434, 0.732416], 0.170730),
        (5, &[0.784398, 0.128625], 0.101751),
        (5, &[0.886579, 0.550053], 0.097878),
        (1, &[0.984125, 0.897439], 0.000000),
      ],
    );
  }

  #[test]
  fn regression_euclidean_medium() {
    assert_regression(
      CentroidEuclideanDistance::new(),
      200,
      3,
      16,
      30,
      30,
      0.0210070595,
      &[
        (1, &[0.054907, 0.254480, 0.692796], 0.000000),
        (8, &[0.079542, 0.757396, 0.880777], 0.197884),
        (8, &[0.172564, 0.390664, 0.430203], 0.168059),
        (12, &[0.178458, 0.143981, 0.605050], 0.250110),
        (4, &[0.186638, 0.766816, 0.429760], 0.054583),
        (9, &[0.199086, 0.500625, 0.772454], 0.223175),
        (3, &[0.210314, 0.653515, 0.055786], 0.032975),
        (5, &[0.217886, 0.901430, 0.184546], 0.073682),
        (4, &[0.323219, 0.167376, 0.205703], 0.105025),
        (5, &[0.353129, 0.118259, 0.847674], 0.106614),
        (14, &[0.416364, 0.805292, 0.567164], 0.323831),
        (7, &[0.458369, 0.625134, 0.301291], 0.135068),
        (3, &[0.476377, 0.536899, 0.072727], 0.028900),
        (6, &[0.494607, 0.835836, 0.851783], 0.111786),
        (6, &[0.501382, 0.208187, 0.382480], 0.087518),
        (9, &[0.531759, 0.201727, 0.662988], 0.301191),
        (6, &[0.573391, 0.911283, 0.191519], 0.142762),
        (6, &[0.575032, 0.170585, 0.101066], 0.081667),
        (4, &[0.583808, 0.451413, 0.807726], 0.044435),
        (5, &[0.666115, 0.876071, 0.474106], 0.055162),
        (6, &[0.734585, 0.738816, 0.841089], 0.180664),
        (9, &[0.779791, 0.505323, 0.389349], 0.215662),
        (10, &[0.781928, 0.580286, 0.692638], 0.185766),
        (4, &[0.784954, 0.551270, 0.067242], 0.065684),
        (17, &[0.805939, 0.258595, 0.830048], 0.547973),
        (8, &[0.843880, 0.227544, 0.233122], 0.109078),
        (6, &[0.861317, 0.821353, 0.079647], 0.097707),
        (9, &[0.907209, 0.106127, 0.413300], 0.201566),
        (4, &[0.943201, 0.864927, 0.451624], 0.043296),
        (2, &[0.960752, 0.848113, 0.607924], 0.029589),
      ],
    );
  }

  #[test]
  fn regression_manhattan_small() {
    assert_regression(
      CentroidManhattanDistance::new(),
      50,
      2,
      10,
      10,
      9,
      0.0174380766,
      &[
        (5, &[0.127111, 0.277504], 0.099242),
        (6, &[0.142299, 0.647631], 0.117171),
        (7, &[0.359254, 0.259409], 0.128684),
        (7, &[0.368375, 0.921907], 0.046654),
        (12, &[0.487616, 0.664971], 0.291177),
        (4, &[0.723951, 0.234193], 0.090809),
        (3, &[0.836218, 0.911816], 0.044151),
        (2, &[0.880585, 0.060295], 0.007922),
        (4, &[0.924478, 0.584008], 0.046093),
      ],
    );
  }

  #[test]
  fn regression_manhattan_medium() {
    assert_regression(
      CentroidManhattanDistance::new(),
      200,
      3,
      16,
      30,
      26,
      0.0293731540,
      &[
        (5, &[0.034950, 0.812707, 0.882723], 0.088729),
        (1, &[0.062279, 0.222807, 0.806743], 0.000000),
        (8, &[0.155831, 0.564768, 0.747961], 0.165287),
        (10, &[0.194615, 0.828946, 0.233846], 0.370744),
        (24, &[0.224847, 0.181920, 0.542109], 1.095942),
        (6, &[0.293250, 0.529138, 0.405022], 0.081712),
        (6, &[0.294046, 0.299751, 0.883665], 0.182069),
        (1, &[0.300418, 0.623529, 0.020696], 0.000000),
        (1, &[0.377142, 0.053507, 0.277326], 0.000000),
        (3, &[0.393664, 0.873392, 0.214621], 0.050769),
        (5, &[0.399981, 0.820544, 0.892738], 0.104281),
        (16, &[0.439781, 0.810411, 0.565492], 0.389890),
        (12, &[0.533977, 0.177879, 0.217281], 0.430178),
        (2, &[0.542570, 0.646867, 0.312191], 0.016113),
        (8, &[0.562968, 0.512812, 0.092820], 0.237145),
        (3, &[0.610740, 0.026155, 0.812243], 0.067055),
        (19, &[0.725792, 0.632336, 0.728099], 0.692684),
        (10, &[0.738689, 0.881069, 0.265596], 0.315693),
        (24, &[0.747537, 0.309583, 0.805573], 0.936733),
        (8, &[0.757605, 0.522628, 0.353230], 0.196291),
        (1, &[0.834655, 0.934109, 0.885711], 0.000000),
        (10, &[0.835564, 0.209939, 0.262411], 0.203521),
        (4, &[0.882105, 0.804818, 0.043160], 0.050135),
        (7, &[0.937184, 0.096587, 0.422938], 0.122505),
        (5, &[0.941087, 0.861724, 0.533732], 0.077155),
        (1, &[0.942394, 0.647449, 0.091448], 0.000000),
      ],
    );
  }

  #[test]
  fn regression_avgintercluster_small() {
    assert_regression(
      AverageInterclusterDistance::new(),
      50,
      2,
      10,
      10,
      7,
      0.0241044560,
      &[
        (4, &[0.093082, 0.660029], 0.073221),
        (12, &[0.262528, 0.266949], 0.386061),
        (16, &[0.356748, 0.774361], 0.413940),
        (6, &[0.667334, 0.690710], 0.094918),
        (5, &[0.784398, 0.128625], 0.101751),
        (2, &[0.854080, 0.940047], 0.037454),
        (5, &[0.886579, 0.550053], 0.097878),
      ],
    );
  }

  #[test]
  fn regression_avgintercluster_medium() {
    assert_regression(
      AverageInterclusterDistance::new(),
      200,
      3,
      16,
      30,
      29,
      0.0228728859,
      &[
        (9, &[0.082830, 0.736705, 0.860207], 0.259948),
        (14, &[0.169825, 0.163367, 0.603150], 0.317902),
        (4, &[0.199345, 0.318578, 0.377564], 0.083335),
        (3, &[0.210314, 0.653515, 0.055786], 0.032975),
        (9, &[0.216942, 0.530905, 0.485369], 0.258542),
        (5, &[0.217886, 0.901430, 0.184546], 0.073682),
        (5, &[0.244340, 0.819742, 0.340941], 0.135870),
        (6, &[0.257846, 0.443455, 0.848652], 0.120184),
        (2, &[0.294050, 0.085437, 0.286131], 0.016003),
        (2, &[0.302272, 0.089406, 0.866609], 0.036125),
        (15, &[0.409211, 0.791123, 0.574256], 0.387300),
        (8, &[0.460197, 0.127353, 0.520091], 0.205953),
        (9, &[0.516403, 0.194765, 0.128299], 0.233483),
        (8, &[0.534476, 0.814003, 0.841832], 0.184803),
        (8, &[0.543580, 0.276412, 0.782842], 0.198613),
        (7, &[0.556951, 0.566100, 0.115960], 0.177423),
        (1, &[0.689603, 0.423824, 0.019880], 0.000000),
        (7, &[0.698418, 0.602148, 0.398909], 0.144704),
        (11, &[0.746083, 0.414986, 0.893126], 0.183679),
        (11, &[0.753522, 0.860139, 0.129108], 0.367737),
        (6, &[0.775552, 0.109954, 0.833348], 0.108105),
        (8, &[0.787545, 0.592843, 0.706621], 0.162138),
        (10, &[0.804264, 0.880733, 0.477327], 0.314754),
        (4, &[0.815310, 0.468314, 0.363582], 0.077041),
        (9, &[0.823363, 0.222978, 0.249847], 0.161027),
        (8, &[0.862670, 0.263834, 0.604332], 0.149145),
        (4, &[0.865971, 0.754941, 0.721558], 0.114823),
        (6, &[0.930736, 0.054306, 0.368129], 0.069284),
        (1, &[0.942394, 0.647449, 0.091448], 0.000000),
      ],
    );
  }

  #[test]
  fn regression_avgintracluster_small() {
    // Basic test: tree builds without panicking and produces reasonable output
    let data = create_data(50, 2);
    let dist = AverageIntraclusterDistance::new();
    let mut tree = CFTree::new(dist.clone(), dist, 10, 2, 10, 0.0);
    for v in data.iter() {
      tree.insert(v);
    }

    // Should have between 5 and 15 leaves for this small dataset
    assert!(
      tree.leaf_entries().len() >= 5 && tree.leaf_entries().len() <= 15,
      "Expected 5-15 leaves, got {}",
      tree.leaf_entries().len()
    );

    // Overall variance should be reasonable
    let ov = overall_variance(&tree);
    assert!(
      ov > 0.01 && ov < 0.1,
      "Overall variance out of expected range: {}",
      ov
    );

    // All leaf entries should have valid statistics
    for cf in tree.leaf_entries() {
      assert!(cf.size() > 0, "Leaf entry should have at least one point");
      assert!(cf.ssd() >= 0.0, "SSD should be non-negative");
      assert_eq!(cf.centroid().len(), 2, "Centroid should have 2 dimensions");
    }
  }
}
