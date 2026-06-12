//! Targeted benchmark for measuring clone/slice optimization impact.
//! Run with: cargo bench --baseline_bench

use betula::cf_tree::CFTree;
use betula::cluster_feature::{ClusterFeature, VII, VVI, VVV};
use betula::distance::{CFDistance, CentroidEuclideanDistance};
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn make_points(n: usize, dim: usize, seed: u64) -> Vec<Vec<f64>> {
  let mut rng = StdRng::seed_from_u64(seed);
  (0..n)
    .map(|_| (0..dim).map(|_| rng.random_range(-100.0..100.0)).collect())
    .collect()
}

// ── CF Tree insert (exercises children() clone path, centroid() usage) ──

fn bench_cf_tree_insert(c: &mut Criterion) {
  let mut group = c.benchmark_group("CFTree_insert");

  for &size in &[1_000, 10_000] {
    let data = make_points(size, 10, 42);

    group.throughput(Throughput::Elements(size as u64));
    group.bench_function(BenchmarkId::from_parameter(size), |b| {
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

// ── CF Tree with splits (exercises the children().clone() hot path) ──

fn bench_cf_tree_split_heavy(c: &mut Criterion) {
  let mut group = c.benchmark_group("CFTree_split_heavy");
  group.measurement_time(std::time::Duration::from_secs(8));

  // Small capacity forces many splits → exercises children().clone() heavily
  let data = make_points(5_000, 10, 42);

  group.throughput(Throughput::Elements(data.len() as u64));
  group.bench_function("cap=4_splits", |b| {
    b.iter(|| {
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
      black_box(tree)
    })
  });
  group.finish();
}

// ── centroid() access pattern (exercises centroid() return type) ──

fn bench_centroid_access(c: &mut Criterion) {
  let mut group = c.benchmark_group("centroid_access");

  for &dim in &[2, 10, 64] {
    let mut cf: VII<f64> = VII::new(dim);
    for p in &make_points(10_000, dim, 1) {
      cf.add(p);
    }

    group.bench_function(format!("VII_centroid_clone dim={}", dim), |b| {
      b.iter(|| {
        // Simulate the pattern: get centroid, clone it, use it
        let centroid = cf.centroid();
        black_box(centroid.iter().sum::<f64>())
      })
    });

    group.bench_function(format!("VII_centroid_borrow dim={}", dim), |b| {
      b.iter(|| {
        // Simulate read-only usage
        let centroid = cf.centroid();
        black_box(centroid.iter().sum::<f64>())
      })
    });
  }
  group.finish();
}

// ── add_cf merge pattern (exercises centroid().clone() in add_cf) ──

fn bench_add_cf_merge(c: &mut Criterion) {
  let mut group = c.benchmark_group("add_cf_merge");

  for &dim in &[2, 10, 64] {
    let mut cf1: VII<f64> = VII::new(dim);
    let mut cf2: VII<f64> = VII::new(dim);
    for p in &make_points(5_000, dim, 1) {
      cf1.add(p);
    }
    for p in &make_points(5_000, dim, 2) {
      cf2.add(p);
    }

    group.bench_function(format!("VII_add_cf dim={}", dim), |b| {
      b.iter(|| {
        let mut merged = cf1.clone();
        merged.add_cf(&cf2);
        black_box(merged)
      })
    });

    let mut cf1_vvi: VVI<f64> = VVI::new(dim);
    let mut cf2_vvi: VVI<f64> = VVI::new(dim);
    for p in &make_points(5_000, dim, 1) {
      cf1_vvi.add(p);
    }
    for p in &make_points(5_000, dim, 2) {
      cf2_vvi.add(p);
    }

    group.bench_function(format!("VVI_add_cf dim={}", dim), |b| {
      b.iter(|| {
        let mut merged = cf1_vvi.clone();
        merged.add_cf(&cf2_vvi);
        black_box(merged)
      })
    });

    let mut cf1_vvv: VVV<f64> = VVV::new(dim);
    let mut cf2_vvv: VVV<f64> = VVV::new(dim);
    for p in &make_points(5_000, dim, 1) {
      cf1_vvv.add(p);
    }
    for p in &make_points(5_000, dim, 2) {
      cf2_vvv.add(p);
    }

    group.bench_function(format!("VVV_add_cf dim={}", dim), |b| {
      b.iter(|| {
        let mut merged = cf1_vvv.clone();
        merged.add_cf(&cf2_vvv);
        black_box(merged)
      })
    });
  }
  group.finish();
}

// ── Distance calculation (exercises centroid() → distance) ──

fn bench_distance_calc(c: &mut Criterion) {
  let mut group = c.benchmark_group("distance_calc");

  for &dim in &[2, 10, 64] {
    let mut cf1: VII<f64> = VII::new(dim);
    let mut cf2: VII<f64> = VII::new(dim);
    for p in &make_points(100, dim, 1) {
      cf1.add(p);
    }
    for p in &make_points(100, dim, 2) {
      cf2.add(p);
    }
    let point = make_points(1, dim, 3)[0].clone();

    let dist = CentroidEuclideanDistance::<f64, VII<f64>>::new();

    group.bench_function(format!("sq_dist_cf dim={}", dim), |b| {
      b.iter(|| {
        let d = dist.sq_dist_cf(&cf1, &cf2, dim);
        black_box(d)
      })
    });

    group.bench_function(format!("sq_dist_point dim={}", dim), |b| {
      b.iter(|| {
        let d = dist.sq_dist(&cf1, &point, dim);
        black_box(d)
      })
    });
  }
  group.finish();
}

// ── ssd_per_dim access pattern ──

fn bench_ssd_per_dim(c: &mut Criterion) {
  let mut group = c.benchmark_group("ssd_per_dim");

  for &dim in &[2, 10, 64] {
    let mut vii: VII<f64> = VII::new(dim);
    let mut vvi: VVI<f64> = VVI::new(dim);
    let mut vvv: VVV<f64> = VVV::new(dim);
    for p in &make_points(10_000, dim, 1) {
      vii.add(p);
      vvi.add(p);
      vvv.add(p);
    }

    group.bench_function(format!("VII_ssd_per_dim dim={}", dim), |b| {
      b.iter(|| {
        let spd = vii.ssd_per_dim();
        black_box(
          spd
            .map(|s| s.iter().copied().sum::<f64>())
            .unwrap_or_default(),
        )
      })
    });

    group.bench_function(format!("VVI_ssd_per_dim dim={}", dim), |b| {
      b.iter(|| {
        let spd = vvi.ssd_per_dim();
        black_box(
          spd
            .map(|s| s.iter().copied().sum::<f64>())
            .unwrap_or_default(),
        )
      })
    });

    group.bench_function(format!("VVV_ssd_per_dim dim={}", dim), |b| {
      b.iter(|| {
        let spd = vvv.ssd_per_dim();
        black_box(
          spd
            .map(|s| s.iter().copied().sum::<f64>())
            .unwrap_or_default(),
        )
      })
    });
  }
  group.finish();
}

// ── children() access pattern ──

fn bench_children_access(c: &mut Criterion) {
  let mut group = c.benchmark_group("children_access");

  let data = make_points(10_000, 10, 42);
  let mut tree: CFTree<f64, VII<f64>, _, _> = CFTree::new(
    CentroidEuclideanDistance::new(),
    CentroidEuclideanDistance::new(),
    16,
    10,
    1000,
    0.0,
  );
  for point in &data {
    tree.insert(point);
  }

  // Simulate the pattern: get children, clone, iterate
  group.bench_function("children_clone_then_iter", |b| {
    b.iter(|| {
      for node in tree.nodes() {
        let childs = node.children();
        black_box(childs.iter().sum::<usize>());
      }
    })
  });

  // Simulate read-only iteration
  group.bench_function("children_borrow_iter", |b| {
    b.iter(|| {
      for node in tree.nodes() {
        let childs = node.children();
        black_box(childs.iter().sum::<usize>());
      }
    })
  });

  group.finish();
}

criterion_group!(
  benches,
  bench_cf_tree_insert,
  bench_cf_tree_split_heavy,
  bench_centroid_access,
  bench_add_cf_merge,
  bench_distance_calc,
  bench_ssd_per_dim,
  bench_children_access,
);
criterion_main!(benches);
