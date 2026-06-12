//! Macro benchmarks for full CF-Tree construction.
//! Run with: cargo run --release --features bench --bin bench

use betula::cf_tree::CFTree;
use betula::cluster_feature::VII;
use betula::distance::CentroidEuclideanDistance;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

fn make_points(n: usize, dim: usize, seed: u64) -> Vec<Vec<f64>> {
  let mut rng = StdRng::seed_from_u64(seed);
  (0..n)
    .map(|_| (0..dim).map(|_| rng.random_range(-100.0..100.0)).collect())
    .collect()
}

fn time_it<F: FnMut()>(mut f: F) -> f64 {
  // Warmup
  for _ in 0..2 {
    f();
  }
  let start = Instant::now();
  let iterations = 5;
  for _ in 0..iterations {
    f();
  }
  start.elapsed().as_secs_f64() / iterations as f64
}

fn main() {
  println!("=== BETULA Macro Benchmarks ===\n");

  // ── 1. CF Tree insert (exercises centroid() + children() paths) ──
  let data = make_points(10_000, 10, 42);
  let t = time_it(|| {
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
    black_box(tree);
  });
  println!("CFTree_insert_10k:          {:.3} ms", t * 1000.0);

  // ── 2. CF Tree with splits (exercises children().clone() hot path) ──
  let data = make_points(5_000, 10, 42);
  let t = time_it(|| {
    let mut tree: CFTree<f64, VII<f64>, _, _> = CFTree::new(
      CentroidEuclideanDistance::new(),
      CentroidEuclideanDistance::new(),
      4,
      10,
      1000,
      0.0,
    );
    for point in &data {
      tree.insert(point);
    }
    black_box(tree);
  });
  println!("CFTree_split_heavy_5k:      {:.3} ms", t * 1000.0);

  println!("\n=== Done ===");
}

#[inline(never)]
fn black_box<T>(val: T) {
  std::hint::black_box(val);
}
