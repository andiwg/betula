use betula::cf_tree::CFTree;
use betula::cluster_feature::{ClusterFeature, VII};
use betula::distance::{
  AverageInterclusterDistance, AverageIntraclusterDistance, CentroidEuclideanDistance,
  CentroidManhattanDistance, RadiusDistance, VarianceIncreaseDistance,
};
use clap::Parser;
use ndarray::Array2;
use npyz::NpyFile;
use serde::{Deserialize, Serialize};
use std::{
  fs::File,
  io::{BufReader, BufWriter, Write},
  time::Instant,
};

#[derive(Parser, Debug)]
#[command(name = "betula-benchmark")]
#[command(about = "Benchmark tool for BETULA CF-Tree implementation")]
struct Args {
  #[arg(long, help = "Path to input numpy file")]
  input: String,

  #[arg(long, help = "Path to output JSON file")]
  output: String,

  #[arg(long, default_value = "32", help = "Tree node capacity")]
  capacity: usize,

  #[arg(long, default_value = "1000", help = "Maximum number of leaf entries")]
  maxleaves: usize,

  #[arg(long, default_value = "0.0", help = "Threshold for cluster splitting")]
  threshold: f64,

  #[arg(
    long,
    default_value = "euclidean",
    help = "Distance measure for insertion: euclidean, manhattan, avgintercluster, avgintracluster, varianceincrease, radius"
  )]
  distance: String,

  #[arg(
    long,
    default_value = "euclidean",
    help = "Distance measure for absorption: euclidean, manhattan, avgintercluster, avgintracluster, varianceincrease, radius"
  )]
  absorption: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct ClusterStats {
  id: usize,
  size: usize,
  centroid: Vec<f64>,
  variance: f64,
}

#[derive(Serialize, Deserialize, Debug)]
struct TimingData {
  load: f64,
  build: f64,
  stats: f64,
  total: f64,
}

#[derive(Serialize, Deserialize, Debug)]
struct Metadata {
  input_file: String,
  num_points: usize,
  dimensions: usize,
  capacity: usize,
  maxleaves: usize,
  num_rebuilds: usize,
}

#[derive(Serialize, Deserialize, Debug)]
struct BenchmarkResults {
  metadata: Metadata,
  timing_ms: TimingData,
  overall_variance: f64,
  clusters: Vec<ClusterStats>,
}

#[derive(Clone, Copy, Debug)]
enum DistanceKind {
  Euclidean,
  Manhattan,
  AvgIntercluster,
  AvgIntracluster,
  VarianceIncrease,
  Radius,
}

impl DistanceKind {
  fn parse(name: &str) -> Option<Self> {
    match name {
      "euclidean" => Some(Self::Euclidean),
      "manhattan" => Some(Self::Manhattan),
      "avgintercluster" => Some(Self::AvgIntercluster),
      "avgintracluster" => Some(Self::AvgIntracluster),
      "varianceincrease" => Some(Self::VarianceIncrease),
      "radius" => Some(Self::Radius),
      _ => None,
    }
  }
}

fn load_numpy(path: &str) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
  let file = File::open(path)?;
  let reader = BufReader::new(file);
  let npy_file = NpyFile::new(reader)?;
  let shape = npy_file.shape().to_vec();

  if shape.len() != 2 {
    return Err("Expected 2D array".into());
  }

  let data: Vec<f64> = npy_file.into_vec()?;
  let array: Array2<f64> = Array2::from_shape_vec((shape[0] as usize, shape[1] as usize), data)?;
  Ok(array)
}

fn write_json(path: &str, results: &BenchmarkResults) -> Result<(), Box<dyn std::error::Error>> {
  let file = File::create(path)?;
  let mut writer = BufWriter::new(file);
  let json = serde_json::to_string_pretty(results)?;
  writer.write_all(json.as_bytes())?;
  Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let args = Args::parse();

  println!("=== BETULA Benchmark ===");
  println!("File: {}", args.input);
  println!(
    "Parameters: capacity={}, maxleaves={}, threshold={}, distance={}, absorption={}",
    args.capacity, args.maxleaves, args.threshold, args.distance, args.absorption
  );
  println!();

  let total_start = Instant::now();

  let load_start = Instant::now();
  let data = load_numpy(&args.input)?;
  let num_points: usize = data.nrows();
  let dimensions: usize = data.ncols();
  let load_time = load_start.elapsed().as_secs_f64() * 1000.0;

  println!("Data points: {}, Dimensions: {}", num_points, dimensions);
  println!();

  // Macro that inlines the full tree-build+stats body for a given
  // (distance, absorption) type pair. D and A are independent — the
  // macro expands to a self-contained block yielding the result tuple.
  macro_rules! run_pair {
    ($D:ty, $A:ty) => {{
      let build_start = Instant::now();
      let mut tree = CFTree::<f64, VII<f64>, $D, $A>::new(
        <$D>::new(),
        <$A>::new(),
        args.capacity,
        dimensions,
        args.maxleaves,
        args.threshold,
      );

      for row in data.rows() {
        let point: &[f64] = row.as_slice().ok_or("row is not contiguous")?;
        // SAFETY: row length == dimensions (guaranteed by the 2D numpy array shape),
        // which matches the tree's expected dimensionality.
        unsafe {
          tree.insert_unchecked(point);
        }
      }
      let build_time = build_start.elapsed().as_secs_f64() * 1000.0;

      let stats_start = Instant::now();
      let leaf_entries = tree.leaf_entries();
      let num_clusters = leaf_entries.len();
      let mut total_variance = 0.0;
      let mut total_points = 0;
      let mut clusters: Vec<ClusterStats> = Vec::new();

      for (id, cf) in leaf_entries.iter().enumerate() {
        let size = cf.size();
        let variance = if size > 0 {
          cf.ssd() / size as f64
        } else {
          0.0
        };
        total_variance += cf.ssd();
        total_points += size;
        clusters.push(ClusterStats {
          id,
          size,
          centroid: cf.centroid().to_vec(),
          variance,
        });
      }

      let overall_variance = if total_points > 0 {
        total_variance / total_points as f64
      } else {
        0.0
      };
      let stats_time = stats_start.elapsed().as_secs_f64() * 1000.0;
      (
        clusters,
        build_time,
        stats_time,
        num_clusters,
        tree.rebuild_count(),
        overall_variance,
      )
    }};
  }

  let distance = DistanceKind::parse(&args.distance).unwrap_or_else(|| {
    eprintln!(
      "Unknown distance: {} (supported: euclidean, manhattan, avgintercluster, avgintracluster, varianceincrease, radius)",
      args.distance
    );
    std::process::exit(1);
  });
  let absorption = DistanceKind::parse(&args.absorption).unwrap_or_else(|| {
    eprintln!(
      "Unknown absorption: {} (supported: euclidean, manhattan, avgintercluster, avgintracluster, varianceincrease, radius)",
      args.absorption
    );
    std::process::exit(1);
  });

  macro_rules! run_absorption {
    ($D:ty, $absorption:expr) => {
      match $absorption {
        DistanceKind::Euclidean => run_pair!($D, CentroidEuclideanDistance<f64, VII<f64>>),
        DistanceKind::Manhattan => run_pair!($D, CentroidManhattanDistance<f64, VII<f64>>),
        DistanceKind::AvgIntercluster => {
          run_pair!($D, AverageInterclusterDistance<f64, VII<f64>>)
        }
        DistanceKind::AvgIntracluster => {
          run_pair!($D, AverageIntraclusterDistance<f64, VII<f64>>)
        }
        DistanceKind::VarianceIncrease => {
          run_pair!($D, VarianceIncreaseDistance<f64, VII<f64>>)
        }
        DistanceKind::Radius => run_pair!($D, RadiusDistance<f64, VII<f64>>),
      }
    };
  }

  let (clusters, build_time, stats_time, num_clusters, rebuild_count, overall_variance) =
    match distance {
      DistanceKind::Euclidean => {
        run_absorption!(CentroidEuclideanDistance<f64, VII<f64>>, absorption)
      }
      DistanceKind::Manhattan => {
        run_absorption!(CentroidManhattanDistance<f64, VII<f64>>, absorption)
      }
      DistanceKind::AvgIntercluster => {
        run_absorption!(AverageInterclusterDistance<f64, VII<f64>>, absorption)
      }
      DistanceKind::AvgIntracluster => {
        run_absorption!(AverageIntraclusterDistance<f64, VII<f64>>, absorption)
      }
      DistanceKind::VarianceIncrease => {
        run_absorption!(VarianceIncreaseDistance<f64, VII<f64>>, absorption)
      }
      DistanceKind::Radius => run_absorption!(RadiusDistance<f64, VII<f64>>, absorption),
    };

  let total_time = total_start.elapsed().as_secs_f64() * 1000.0;

  println!("Load time:      {:>10.2} ms", load_time);
  println!("Build time:      {:>10.2} ms", build_time);
  println!("Stats time:     {:>10.2} ms", stats_time);
  println!("Total time:     {:>10.2} ms", total_time);
  println!();

  println!("Number of rebuilds: {}", rebuild_count);
  println!("Number of clusters: {}", num_clusters);
  println!("Overall variance:   {:.6}", overall_variance);
  println!();

  let results = BenchmarkResults {
    metadata: Metadata {
      input_file: args.input.clone(),
      num_points,
      dimensions,
      capacity: args.capacity,
      maxleaves: args.maxleaves,
      num_rebuilds: rebuild_count,
    },
    timing_ms: TimingData {
      load: load_time,
      build: build_time,
      stats: stats_time,
      total: total_time,
    },
    overall_variance,
    clusters,
  };

  write_json(&args.output, &results)?;
  println!("Results written to: {}", args.output);

  Ok(())
}
