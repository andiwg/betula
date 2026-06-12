use betula::cf_tree::CFTree;
use betula::cluster_feature::VII;
use betula::distance::CentroidEuclideanDistance;
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn bench_insert(c: &mut Criterion) {
  let mut group = c.benchmark_group("CFTree insert");

  for &size in &[1_000, 10_000, 100_000] {
    let mut rng = StdRng::seed_from_u64(42);
    let data: Vec<Vec<f64>> = (0..size)
      .map(|_| vec![rng.random_range(-100.0..100.0); 10])
      .collect();

    group.bench_function(format!("{} points", size), |b| {
      b.iter(|| {
        let mut tree: CFTree<f64, VII<f64>, _, _> = CFTree::new(
          CentroidEuclideanDistance::new(),
          CentroidEuclideanDistance::new(),
          32,
          10,
          1000,
          0.0,
        );
        for point in &data {
          tree.insert(point);
        }
        black_box(tree)
      })
    });
  }

  group.finish();
}

criterion_group!(benches, bench_insert);
criterion_main!(benches);
