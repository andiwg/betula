use betula::cluster_feature::{ClusterFeature, VII, VVV};
use betula::distance::{
  AverageInterclusterDistance, AverageIntraclusterDistance, CFDistance, CentroidEuclideanDistance,
  CentroidManhattanDistance, Manhattan, RadiusDistance, SqEuclidean, VarianceIncreaseDistance,
  VectorDistance,
};
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn make_points(n: usize, dim: usize, seed: u64) -> Vec<Vec<f64>> {
  let mut rng = StdRng::seed_from_u64(seed);
  (0..n)
    .map(|_| (0..dim).map(|_| rng.random_range(-100.0..100.0)).collect())
    .collect()
}

fn bench_centroid_distance(c: &mut Criterion) {
  let mut group = c.benchmark_group("Centroid distance");

  for &dim in &[2, 10, 64] {
    let mut cf1: VII<f64> = VII::new(dim);
    for p in &make_points(100, dim, 1) {
      cf1.add(p);
    }
    let mut cf2: VII<f64> = VII::new(dim);
    for p in &make_points(100, dim, 2) {
      cf2.add(p);
    }
    let point = make_points(1, dim, 3)[0].clone();

    group.bench_function(format!("dim={}", dim), |b| {
      b.iter(|| {
        let dist = CentroidEuclideanDistance::new();
        let _ = dist.sq_dist(&cf1, &point, dim) + dist.sq_dist_cf(&cf1, &cf2, dim);
        black_box(())
      })
    });
  }

  group.finish();
}

fn bench_distance_measures(c: &mut Criterion) {
  let mut group = c.benchmark_group("Distance measures");

  let dim = 10;
  let mut cf1: VII<f64> = VII::new(dim);
  for p in &make_points(100, dim, 1) {
    cf1.add(p);
  }
  let mut cf2: VII<f64> = VII::new(dim);
  for p in &make_points(100, dim, 2) {
    cf2.add(p);
  }
  let point = make_points(1, dim, 3)[0].clone();

  let measures: Vec<(&str, Box<dyn Fn() -> f64>)> = vec![
    (
      "CentroidEuclidean",
      Box::new(|| {
        let d = CentroidEuclideanDistance::new();
        d.sq_dist(&cf1, &point, dim) + d.sq_dist_cf(&cf1, &cf2, dim)
      }),
    ),
    (
      "CentroidManhattan",
      Box::new(|| {
        let d = CentroidManhattanDistance::new();
        d.sq_dist(&cf1, &point, dim) + d.sq_dist_cf(&cf1, &cf2, dim)
      }),
    ),
    (
      "AverageIntercluster",
      Box::new(|| {
        let d = AverageInterclusterDistance::new();
        d.sq_dist(&cf1, &point, dim) + d.sq_dist_cf(&cf1, &cf2, dim)
      }),
    ),
    (
      "AverageIntracluster",
      Box::new(|| {
        let d = AverageIntraclusterDistance::new();
        d.sq_dist(&cf1, &point, dim) + d.sq_dist_cf(&cf1, &cf2, dim)
      }),
    ),
    (
      "VarianceIncrease",
      Box::new(|| {
        let d = VarianceIncreaseDistance::new();
        d.sq_dist(&cf1, &point, dim) + d.sq_dist_cf(&cf1, &cf2, dim)
      }),
    ),
    (
      "Radius",
      Box::new(|| {
        let d = RadiusDistance::new();
        d.sq_dist(&cf1, &point, dim) + d.sq_dist_cf(&cf1, &cf2, dim)
      }),
    ),
  ];

  for (name, bench_fn) in measures {
    group.bench_function(name, |b| {
      b.iter(|| {
        let result = bench_fn();
        black_box(result)
      })
    });
  }

  group.finish();
}

fn bench_covariance(c: &mut Criterion) {
  let mut group = c.benchmark_group("Covariance");

  for &dim in &[2, 5, 10] {
    let mut vvv: VVV<f64> = VVV::new(dim);
    for p in &make_points(100, dim, 1) {
      vvv.add(p);
    }

    group.bench_function(format!("dim={}", dim), |b| {
      b.iter(|| {
        let cov = vvv.covariance().unwrap();
        black_box(cov)
      })
    });
  }

  group.finish();
}

fn bench_vector_distance(c: &mut Criterion) {
  let mut group = c.benchmark_group("VectorDistance");

  for &dim in &[2, 10, 64, 256] {
    let va: Vec<f64> = (0..dim).map(|i| (i as f64) * 0.1).collect();
    let vb: Vec<f64> = (0..dim).map(|i| ((i as f64) * 0.1).sin().abs()).collect();

    group.bench_function(format!("SqEuclidean dim={}", dim), |bencher| {
      bencher.iter(|| {
        let d = SqEuclidean::dist::<f64>(&va, &vb, dim);
        black_box(d)
      })
    });

    group.bench_function(format!("Manhattan dim={}", dim), |bencher| {
      bencher.iter(|| {
        let d = Manhattan::dist::<f64>(&va, &vb, dim);
        black_box(d)
      })
    });
  }

  group.finish();
}

criterion_group!(
  benches,
  bench_centroid_distance,
  bench_distance_measures,
  bench_covariance,
  bench_vector_distance
);
criterion_main!(benches);
