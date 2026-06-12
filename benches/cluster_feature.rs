use betula::cluster_feature::{ClusterFeature, VII, VVI, VVV};
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn make_points(n: usize, dim: usize, seed: u64) -> Vec<Vec<f64>> {
  let mut rng = StdRng::seed_from_u64(seed);
  (0..n)
    .map(|_| (0..dim).map(|_| rng.random_range(-100.0..100.0)).collect())
    .collect()
}

fn bench_add(c: &mut Criterion) {
  let mut group = c.benchmark_group("ClusterFeature::add");

  for &dim in &[2, 10, 64] {
    let points = make_points(10_000, dim, 1);

    group.bench_function(format!("VII dim={}", dim), |b| {
      b.iter(|| {
        let mut cf: VII<f64> = VII::new(dim);
        for p in &points {
          cf.add(p);
        }
        black_box(cf)
      })
    });

    group.bench_function(format!("VVI dim={}", dim), |b| {
      b.iter(|| {
        let mut cf: VVI<f64> = VVI::new(dim);
        for p in &points {
          cf.add(p);
        }
        black_box(cf)
      })
    });

    group.bench_function(format!("VVV dim={}", dim), |b| {
      b.iter(|| {
        let mut cf: VVV<f64> = VVV::new(dim);
        for p in &points {
          cf.add(p);
        }
        black_box(cf)
      })
    });
  }

  group.finish();
}

fn bench_add_cf(c: &mut Criterion) {
  let mut group = c.benchmark_group("ClusterFeature::add_cf");

  for &dim in &[2, 10, 64] {
    let points1: Vec<Vec<f64>> = make_points(5_000, dim, 1);
    let points2: Vec<Vec<f64>> = make_points(5_000, dim, 2);

    let mut cf1_vii: VII<f64> = VII::new(dim);
    for p in &points1 {
      cf1_vii.add(p);
    }
    let mut cf2_vii: VII<f64> = VII::new(dim);
    for p in &points2 {
      cf2_vii.add(p);
    }

    let mut cf1_vvi: VVI<f64> = VVI::new(dim);
    for p in &points1 {
      cf1_vvi.add(p);
    }
    let mut cf2_vvi: VVI<f64> = VVI::new(dim);
    for p in &points2 {
      cf2_vvi.add(p);
    }

    let mut cf1_vvv: VVV<f64> = VVV::new(dim);
    for p in &points1 {
      cf1_vvv.add(p);
    }
    let mut cf2_vvv: VVV<f64> = VVV::new(dim);
    for p in &points2 {
      cf2_vvv.add(p);
    }

    group.bench_function(format!("VII dim={}", dim), |b| {
      b.iter(|| {
        let mut merged = cf1_vii.clone();
        merged.add_cf(&cf2_vii);
        black_box(merged)
      })
    });

    group.bench_function(format!("VVI dim={}", dim), |b| {
      b.iter(|| {
        let mut merged = cf1_vvi.clone();
        merged.add_cf(&cf2_vvi);
        black_box(merged)
      })
    });

    group.bench_function(format!("VVV dim={}", dim), |b| {
      b.iter(|| {
        let mut merged = cf1_vvv.clone();
        merged.add_cf(&cf2_vvv);
        black_box(merged)
      })
    });
  }

  group.finish();
}

fn bench_query(c: &mut Criterion) {
  let mut group = c.benchmark_group("ClusterFeature queries");

  for &dim in &[2, 10, 64] {
    let mut vii: VII<f64> = VII::new(dim);
    for p in &make_points(10_000, dim, 1) {
      vii.add(p);
    }
    let mut vvi: VVI<f64> = VVI::new(dim);
    for p in &make_points(10_000, dim, 2) {
      vvi.add(p);
    }
    let mut vvv: VVV<f64> = VVV::new(dim);
    for p in &make_points(10_000, dim, 3) {
      vvv.add(p);
    }

    group.bench_function(format!("VII::ssd dim={}", dim), |b| {
      b.iter(|| black_box(vii.ssd()))
    });
    group.bench_function(format!("VII::variance dim={}", dim), |b| {
      b.iter(|| black_box(vii.variance(0)))
    });
    group.bench_function(format!("VII::centroid dim={}", dim), |b| {
      b.iter(|| black_box(vii.centroid()))
    });

    group.bench_function(format!("VVI::ssd dim={}", dim), |b| {
      b.iter(|| black_box(vvi.ssd()))
    });
    group.bench_function(format!("VVI::variance dim={}", dim), |b| {
      b.iter(|| black_box(vvi.variance(0)))
    });
    group.bench_function(format!("VVI::centroid dim={}", dim), |b| {
      b.iter(|| black_box(vvi.centroid()))
    });

    group.bench_function(format!("VVV::ssd dim={}", dim), |b| {
      b.iter(|| black_box(vvv.ssd()))
    });
    group.bench_function(format!("VVV::variance dim={}", dim), |b| {
      b.iter(|| black_box(vvv.variance(0)))
    });
    group.bench_function(format!("VVV::centroid dim={}", dim), |b| {
      b.iter(|| black_box(vvv.centroid()))
    });
    group.bench_function(format!("VVV::covariance dim={}", dim), |b| {
      b.iter(|| {
        let cov = vvv.covariance().unwrap();
        black_box(cov)
      })
    });
  }

  group.finish();
}

fn bench_first_insert(c: &mut Criterion) {
  let mut group = c.benchmark_group("ClusterFeature::first_insert");

  for &dim in &[2, 10, 64] {
    let points = make_points(1, dim, 1);

    group.bench_function(format!("VII dim={}", dim), |b| {
      b.iter(|| {
        let mut cf: VII<f64> = VII::new(dim);
        cf.add(&points[0]);
        black_box(cf)
      })
    });

    group.bench_function(format!("VVI dim={}", dim), |b| {
      b.iter(|| {
        let mut cf: VVI<f64> = VVI::new(dim);
        cf.add(&points[0]);
        black_box(cf)
      })
    });

    group.bench_function(format!("VVV dim={}", dim), |b| {
      b.iter(|| {
        let mut cf: VVV<f64> = VVV::new(dim);
        cf.add(&points[0]);
        black_box(cf)
      })
    });
  }

  group.finish();
}

criterion_group!(
  benches,
  bench_add,
  bench_add_cf,
  bench_query,
  bench_first_insert
);
criterion_main!(benches);
