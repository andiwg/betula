#!/usr/bin/env python3
"""
Comparison script for BETULA Rust vs Java implementations.

Generates test datasets, runs both implementations with various parameters,
and compares leaf node quality (variance) and number of leaves.

ELKI uses NumpyDatabaseConnection for .npy files directly.

Usage:
    python3 verify_comparison.py [--data-dir DIR] [--java-jar JAR] [--rust-bin BIN]

Parameters tested:
    capacity: 8, 16, 32, 64
    maxleaves: 50, 100, 200, 500

Default:
    data-dir: ./data
    java-jar: ./misc/elki/elki-bundle-0.8.1-SNAPSHOT.jar
    rust-bin: ./target/release/betula-benchmark
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path

# Resolve defaults relative to this script's directory
_SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_DATA_DIR = str(_SCRIPT_DIR / "data")

# ELKI bundle JAR — not shipped in the repo (AGPLv3 license)
# Or pass --java-jar explicitly.
DEFAULT_ELKI_JAR = str(_SCRIPT_DIR / "misc" / "elki" / "elki-bundle-0.8.1-SNAPSHOT.jar")
DEFAULT_RUST_BIN = str(_SCRIPT_DIR / ".." / "target" / "release" / "betula")

PARAMS = {
    "capacity": [8, 16, 32, 64],
    "maxleaves": [5000, 10000, 15000],
}


def get_params_for_size(n_points):
    """Get appropriate parameters based on dataset size."""
    if n_points <= 10000:
        return {
            "capacity": [8, 16, 32, 64],
            "maxleaves": [100, 200, 500],
        }
    elif n_points <= 50000:
        return {
            "capacity": [8, 16, 32, 64],
            "maxleaves": [1000, 2000, 5000],
        }
    else:
        return PARAMS


# Distance configuration: maps CLI labels to ELKI class names.
# ELKI class names verified against elki-bundle jar contents.
DIST_CONFIGS = [
    {
        "distance": "euclidean",
        "absorption": "euclidean",
        "label": "euclidean",
        "distance_class": "CentroidEuclideanDistance",
        "absorption_class": "CentroidEuclideanDistance",
    },
    {
        "distance": "manhattan",
        "absorption": "manhattan",
        "label": "manhattan",
        "distance_class": "CentroidManhattanDistance",
        "absorption_class": "CentroidManhattanDistance",
    },
    {
        "distance": "avgintercluster",
        "absorption": "avgintercluster",
        "label": "avgintercluster",
        "distance_class": "AverageInterclusterDistance",
        "absorption_class": "AverageInterclusterDistance",
    },
    {
        "distance": "avgintracluster",
        "absorption": "avgintracluster",
        "label": "avgintracluster",
        "distance_class": "AverageIntraclusterDistance",
        "absorption_class": "AverageIntraclusterDistance",
    },
    {
        "distance": "varianceincrease",
        "absorption": "varianceincrease",
        "label": "varianceincrease",
        "distance_class": "VarianceIncreaseDistance",
        "absorption_class": "VarianceIncreaseDistance",
    },
    {
        "distance": "radius",
        "absorption": "radius",
        "label": "radius",
        "distance_class": "RadiusDistance",
        "absorption_class": "RadiusDistance",
    },
]


def generate_data(data_dir, n_points, n_dims, seed, n_datasets=3):
    """Generate test datasets using Python numpy."""
    import numpy as np

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    for i in range(n_datasets):
        fname = f"dataset_{seed}_{i}.npy"
        fpath = data_path / fname
        if fpath.exists():
            continue
        rng = np.random.default_rng(seed * 1000 + i)
        data = rng.uniform(-100.0, 100.0, size=(n_points, n_dims))
        np.save(str(fpath), data)
        print(f"  Generated {n_points} x {n_dims} array -> {fpath}")


def run_rust_benchmark(
    npy_path, capacity, maxleaves, output_path, dist_label="euclidean"
):
    """Run Rust benchmark tool. Returns (result, build_ms).

    Uses the Rust benchmark's own timing_ms.build value (excludes dataset load time).
    """
    result = subprocess.run(
        [
            DEFAULT_RUST_BIN,
            "--input",
            str(npy_path),
            "--output",
            str(output_path),
            "--capacity",
            str(capacity),
            "--maxleaves",
            str(maxleaves),
            "--distance",
            dist_label,
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    # Extract build time from JSON output
    build_ms = None
    if result.returncode == 0 and Path(output_path).exists():
        try:
            with open(output_path) as f:
                data = json.load(f)
            build_ms = data.get("timing_ms", {}).get("build")
        except (json.JSONDecodeError, KeyError):
            pass
    return result, build_ms


def run_java_betula(
    npy_path, capacity, maxleaves, output_path, dist_config, jar_path=None
):
    """Run Java BETULA via ELKI CLI with NumpyDatabaseConnection.

    ELKI outputs results to stdout, so we capture stdout and write to file.
    Returns (result, build_ms, runtime_ms, load_ms) parsed from Java's log output.

    We prefer ELKI's CFTree.buildtime for fair comparison with Rust's
    timing_ms.build, which excludes dataset load and result extraction.
    """
    if jar_path is None:
        jar_path = DEFAULT_ELKI_JAR
    cmd = [
        "java",
        "-jar",
        jar_path,
        "-dbc",
        "NumpyDatabaseConnection",
        "-dbc.in",
        str(npy_path),
        "-time",
        "-algorithm",
        "elki.clustering.BetulaLeafPreClustering",
        "-cftree.branching",
        str(capacity),
        "-cftree.maxleaves",
        str(maxleaves),
        "-cftree.features",
        "VIIFeature",
        "-cftree.threshold.heuristic",
        "MEAN",
        "-betula.storeids",
    ]

    # Add distance/absorption parameters using verified ELKI class names
    cmd.extend(["-cftree.distance", dist_config["distance_class"]])
    cmd.extend(["-cftree.absorption", dist_config["absorption_class"]])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    # Write stdout (which contains the cluster output) to file
    with open(output_path, "w") as f:
        f.write(result.stdout)

    # Parse timings from Java's log output
    build_ms = None
    runtime_ms = None
    load_ms = None
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            if "elki.index.tree.betula.CFTree.buildtime:" in line:
                try:
                    build_ms = float(
                        line.split(":")[-1].strip().replace(" ms", "").replace("ms", "")
                    )
                except ValueError:
                    pass
            elif "elki.clustering.BetulaLeafPreClustering.runtime:" in line:
                try:
                    runtime_ms = float(
                        line.split(":")[-1].strip().replace(" ms", "").replace("ms", "")
                    )
                except ValueError:
                    pass
            elif "elki.datasource.NumpyDatabaseConnection.loadtime:" in line:
                try:
                    load_ms = float(
                        line.split(":")[-1].strip().replace(" ms", "").replace("ms", "")
                    )
                except ValueError:
                    pass

    return result, build_ms, runtime_ms, load_ms


def parse_elki_output(output_file, n_dims):
    """Parse ELKI verbose output to extract cluster statistics."""
    clusters = []

    if not os.path.exists(output_file):
        return clusters

    with open(output_file, "r") as f:
        content = f.read()

    # Parse clusters from output
    # Format:
    # # Cluster: Cluster N
    # # Cluster name: Cluster
    # # Cluster noise flag: false
    # # Cluster size: M
    # # Model class: elki.data.model.EMModel
    # # Cluster Mean: x, y, ...
    # # weight: M.0
    # # Covariance Matrix: [
    # #  [[v]]
    # # ]

    cluster_pattern = re.compile(
        r"# Cluster: Cluster (\d+).*?"
        r"# Cluster size: (\d+).*?"
        r"# Cluster Mean: (.+?)\s*"
        r"# weight: ([\d.]+).*?"
        r"# Covariance Matrix: \[.*?\[([\d.eE+-]+)\].*?\]",
        re.DOTALL,
    )

    for m in cluster_pattern.finditer(content):
        cluster_id = int(m.group(1))
        size = int(m.group(2))
        centroid_str = m.group(3).strip()
        weight = float(m.group(4))
        cov_val = float(m.group(5))

        # Parse centroid
        centroid = [float(x.strip()) for x in centroid_str.split(",")]

        # For VIIFeature: covariance[0][0] = ssd / (dim * n)
        # So: ssd = cov * dim * n, variance = ssd / n = cov * dim
        ssd = cov_val * n_dims * size
        variance = ssd / size if size > 0 else 0.0

        clusters.append(
            {
                "id": cluster_id,
                "size": size,
                "centroid": centroid,
                "ssd": ssd,
                "variance": variance,
            }
        )

    return clusters


def load_json(path):
    """Load JSON results file."""
    with open(path, "r") as f:
        return json.load(f)


def compare_results(rust_results, java_clusters, n_dims, label):
    """Compare Rust and Java results."""
    rust_clusters = rust_results["clusters"]

    rust_num_leaves = len(rust_clusters)
    java_num_leaves = len(java_clusters)

    rust_overall_var = rust_results["overall_variance"]
    java_overall_var = (
        sum(c["variance"] * c["size"] for c in java_clusters)
        / sum(c["size"] for c in java_clusters)
        if java_clusters
        else 0.0
    )
    # Rust clusters may not have ssd, compute from variance * size
    rust_overall_ssd = sum(
        c.get("ssd", c["variance"] * c["size"]) for c in rust_clusters
    )
    java_overall_ssd = sum(
        c.get("ssd", c["variance"] * c["size"]) for c in java_clusters
    )

    # Sort clusters by size for comparison
    rust_sorted = sorted(rust_clusters, key=lambda c: c["size"], reverse=True)
    java_sorted = sorted(java_clusters, key=lambda c: c["size"], reverse=True)

    n_compare = min(len(rust_sorted), len(java_sorted), 10)

    rust_top = rust_sorted[:n_compare]
    java_top = java_sorted[:n_compare]

    # Ensure ssd is available for all clusters
    for c in rust_top:
        if "ssd" not in c:
            c["ssd"] = c["variance"] * c["size"]
    for c in java_top:
        if "ssd" not in c:
            c["ssd"] = c["variance"] * c["size"]

    # Compute pass/fail criteria
    comparison = {
        "leaves_diff": rust_num_leaves - java_num_leaves,
        "variance_ratio": rust_overall_var / java_overall_var
        if java_overall_var > 0
        else float("inf"),
        "ssd_ratio": rust_overall_ssd / java_overall_ssd
        if java_overall_ssd > 0
        else float("inf"),
        "points_match": sum(c["size"] for c in rust_clusters)
        == sum(c["size"] for c in java_clusters),
    }

    # Pass/fail thresholds: variance ratio and SSD ratio within 5%
    comparison["pass"] = (
        0.95 <= comparison["variance_ratio"] <= 1.05
        and 0.95 <= comparison["ssd_ratio"] <= 1.05
    )

    return {
        "label": label,
        "rust": {
            "num_leaves": rust_num_leaves,
            "overall_variance": rust_overall_var,
            "overall_ssd": rust_overall_ssd,
            "total_points": sum(c["size"] for c in rust_clusters),
            "cluster_sizes": [c["size"] for c in rust_top],
            "cluster_variances": [c["variance"] for c in rust_top],
            "cluster_ssds": [c["ssd"] for c in rust_top],
        },
        "java": {
            "num_leaves": java_num_leaves,
            "overall_variance": java_overall_var,
            "overall_ssd": java_overall_ssd,
            "total_points": sum(c["size"] for c in java_clusters),
            "cluster_sizes": [c["size"] for c in java_top],
            "cluster_variances": [c["variance"] for c in java_top],
            "cluster_ssds": [c["ssd"] for c in java_top],
        },
        "comparison": comparison,
    }


def print_comparison(comp, dataset_name):
    """Print comparison results."""
    print(f"\n{'=' * 70}")
    print(f"Dataset: {dataset_name} ({comp['label']})")
    print(f"{'=' * 70}")

    rust = comp["rust"]
    java = comp["java"]
    cmp = comp["comparison"]

    print(
        f"\n  Leaves:      Rust={rust['num_leaves']:5d}  Java={java['num_leaves']:5d}  Diff={cmp['leaves_diff']:+d}"
    )
    print(
        f"  Points:      Rust={rust['total_points']:5d}  Java={java['total_points']:5d}  Match={cmp['points_match']}"
    )
    print(
        f"  Overall Var: Rust={rust['overall_variance']:.6f}  Java={java['overall_variance']:.6f}  Ratio={cmp['variance_ratio']:.4f}"
    )
    print(
        f"  Overall SSD: Rust={rust['overall_ssd']:.6f}  Java={java['overall_ssd']:.6f}  Ratio={cmp['ssd_ratio']:.4f}"
    )

    print(f"\n  Top clusters (by size):")
    print(
        f"  {'#':>3} {'R-Size':>7} {'J-Size':>7} {'R-Var':>10} {'J-Var':>10} {'R-SSD':>10} {'J-SSD':>10}"
    )
    for i in range(min(len(rust["cluster_sizes"]), len(java["cluster_sizes"]))):
        print(
            f"  {i + 1:3d} {rust['cluster_sizes'][i]:7d} {java['cluster_sizes'][i]:7d} "
            f"{rust['cluster_variances'][i]:10.6f} {java['cluster_variances'][i]:10.6f} "
            f"{rust['cluster_ssds'][i]:10.6f} {java['cluster_ssds'][i]:10.6f}"
        )


def main():
    # ── Pre-flight check: ELKI JAR ──
    # We check the default JAR path; if the user passes --java-jar we skip this.
    default_jar = Path(DEFAULT_ELKI_JAR)
    if not default_jar.exists():
        print("=" * 70)
        print("WARNING: ELKI bundle JAR not found at default location.")
        print("=" * 70)
        print()
        print("The ELKI JAR is not shipped in the repository (AGPLv3 license).")
        print()
        print("Then re-run this script (optionally with --java-jar /path/to/jar).")
        print("=" * 70)
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Compare BETULA Rust vs Java implementations"
    )
    parser.add_argument(
        "--data-dir", default=DEFAULT_DATA_DIR, help="Directory for test data"
    )
    parser.add_argument(
        "--rust-bin", default=DEFAULT_RUST_BIN, help="Path to Rust benchmark binary"
    )
    parser.add_argument(
        "--java-jar", default=DEFAULT_ELKI_JAR, help="Path to ELKI JAR file"
    )
    parser.add_argument(
        "--output-results",
        default=None,
        help="Path to save full comparison results as JSON",
    )
    parser.add_argument(
        "--n-points",
        default=75000,
        type=int,
        help="Number of data points (50k-100k range)",
    )
    parser.add_argument("--n-dims", default=10, type=int, help="Number of dimensions")
    parser.add_argument(
        "--n-datasets", default=3, type=int, help="Number of datasets per seed"
    )
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument(
        "--no-gen-data",
        action="store_true",
        help="Skip dataset generation (use existing datasets only)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print("=" * 70)
    print("BETULA Rust vs Java Comparison")
    print("=" * 70)
    print(
        f"Data: {args.n_points} points x {args.n_dims} dims, {args.n_datasets} datasets"
    )
    print(f"Seeds: {args.seed}")
    print(f"Params: capacity={PARAMS['capacity']}, maxleaves={PARAMS['maxleaves']}")
    print(f"Distance configs: {[c['label'] for c in DIST_CONFIGS]}")
    print()

    # Generate datasets (only if not skipping and datasets don't exist)
    if not args.no_gen_data:
        npy_files_before = sorted(data_dir.glob("dataset_*.npy"))
        if not npy_files_before:
            print("Generating datasets...")
            generate_data(
                data_dir, args.n_points, args.n_dims, args.seed, args.n_datasets
            )
        else:
            print(
                f"Found {len(npy_files_before)} existing datasets, skipping generation"
            )
    else:
        print("Dataset generation skipped (--no-gen-data)")

    # Get list of dataset files
    npy_files = sorted(data_dir.glob("dataset_*.npy"))
    if not npy_files:
        print("No dataset files found!")
        sys.exit(1)

    print(f"\nFound {len(npy_files)} datasets")

    # Run comparison for each dataset and parameter combination
    all_results = []

    for npy_file in npy_files:
        dataset_name = npy_file.stem
        # Get dataset size for parameter selection
        import numpy as np

        data = np.load(str(npy_file))
        n_points = data.shape[0]
        params = get_params_for_size(n_points)

        print(f"\n{'=' * 70}")
        print(f"Processing: {dataset_name} ({n_points} points)")
        print(f"Params: capacity={params['capacity']}, maxleaves={params['maxleaves']}")
        print(f"{'=' * 70}")

        for dist_config in DIST_CONFIGS:
            dist_label = dist_config["label"]
            print(
                f"\n  Distance config: {dist_label} ({dist_config['distance']}/{dist_config['absorption']})"
            )

            for capacity, maxleaves in product(params["capacity"], params["maxleaves"]):
                # Run Rust
                rust_output = (
                    data_dir
                    / f"{dataset_name}_rust_{dist_label}_c{capacity}_ml{maxleaves}.json"
                )
                rust_result, rust_time = run_rust_benchmark(
                    npy_file, capacity, maxleaves, rust_output, dist_label
                )

                if rust_result.returncode != 0:
                    print(
                        f"    Rust FAILED (c={capacity}, ml={maxleaves}): {rust_result.stderr[:200]}"
                    )
                    continue

                # Run Java via ELKI CLI
                java_output = (
                    data_dir
                    / f"{dataset_name}_java_{dist_label}_c{capacity}_ml{maxleaves}.txt"
                )
                java_result, java_build_time, java_runtime_time, java_load_time = (
                    run_java_betula(
                        npy_file,
                        capacity,
                        maxleaves,
                        java_output,
                        dist_config,
                        jar_path=args.java_jar,
                    )
                )

                if java_result.returncode != 0:
                    print(
                        f"    Java FAILED (c={capacity}, ml={maxleaves}): {java_result.stderr[:200]}"
                    )
                    continue

                # Parse Java output
                java_clusters = parse_elki_output(java_output, args.n_dims)

                if not java_clusters:
                    print(f"    Java: No clusters parsed from output")
                    continue

                # Load Rust results
                rust_data = load_json(rust_output)

                # Compare
                comp = compare_results(
                    rust_data,
                    java_clusters,
                    args.n_dims,
                    f"{dist_label} c={capacity} ml={maxleaves}",
                )
                comp["rust_build_time_ms"] = rust_time
                comp["java_build_time_ms"] = java_build_time
                comp["java_runtime_time_ms"] = java_runtime_time
                comp["java_load_time_ms"] = java_load_time
                all_results.append(comp)

                # Print summary
                java_build_str = (
                    f"{java_build_time:7.1f}ms"
                    if java_build_time is not None
                    else "   n/a  "
                )
                java_runtime_str = (
                    f"{java_runtime_time:7.1f}ms"
                    if java_runtime_time is not None
                    else "   n/a  "
                )
                print(
                    f"    c={capacity:3d} ml={maxleaves:4d}: "
                    f"R-leaves={comp['rust']['num_leaves']:4d} J-leaves={comp['java']['num_leaves']:4d} "
                    f"R-var={comp['rust']['overall_variance']:.6f} J-var={comp['java']['overall_variance']:.6f} "
                    f"var-ratio={comp['comparison']['variance_ratio']:.4f} "
                    f"R-build={rust_time:7.1f}ms J-build={java_build_str} J-run={java_runtime_str}"
                )

    # Print summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY TABLE")
    print(f"{'=' * 70}")

    total_pass = 0
    total_fail = 0

    for dist_config in DIST_CONFIGS:
        label = dist_config["label"]
        configs = [r for r in all_results if label in r["label"]]

        if not configs:
            continue

        print(f"\nDistance: {label}")
        print(
            f"{'Capacity':>8} {'MaxLeaves':>9} {'R-Leaves':>9} {'J-Leaves':>9} {'R-Var':>12} {'J-Var':>12} {'VarRatio':>9} {'SSDRatio':>9} {'R-Build':>10} {'J-Build':>10} {'J-Run':>10} {'PtsMatch':>9} {'Status':>7}"
        )

        for c in configs:
            r = c["rust"]
            j = c["java"]
            cmp = c["comparison"]
            cap = c["label"].split("c=")[1].split(" ")[0]
            ml = c["label"].split("ml=")[1].split(" ")[0]
            java_build = c.get("java_build_time_ms")
            java_runtime = c.get("java_runtime_time_ms")
            java_build_str = (
                f"{java_build:10.1f}" if java_build is not None else f"{'n/a':>10}"
            )
            java_runtime_str = (
                f"{java_runtime:10.1f}" if java_runtime is not None else f"{'n/a':>10}"
            )
            status = "PASS" if cmp["pass"] else "FAIL"
            if cmp["pass"]:
                total_pass += 1
            else:
                total_fail += 1
            print(
                f"{cap:>8} {ml:>9} {r['num_leaves']:9d} {j['num_leaves']:9d} "
                f"{r['overall_variance']:12.6f} {j['overall_variance']:12.6f} "
                f"{cmp['variance_ratio']:9.4f} {cmp['ssd_ratio']:9.4f} "
                f"{c['rust_build_time_ms']:10.1f} {java_build_str} {java_runtime_str} "
                f"{'Yes' if cmp['points_match'] else 'NO':>9} {status:>7}"
            )

    print(f"\n{'=' * 70}")
    print("PASS/FAIL SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Passed: {total_pass}")
    print(f"  Failed: {total_fail}")
    print(f"  Total:  {total_pass + total_fail}")
    if total_fail > 0:
        print(
            f"\n  Note: Thresholds are variance_ratio and ssd_ratio within [0.95, 1.05]."
        )
    print(f"\nComparison complete!")
    print(f"{'=' * 70}")

    # Save full results to JSON if requested
    if args.output_results:
        output_path = Path(args.output_results)
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "n_points": args.n_points,
                "n_dims": args.n_dims,
                "n_datasets": args.n_datasets,
                "seed": args.seed,
                "data_dir": str(data_dir),
                "rust_bin": args.rust_bin,
                "java_jar": args.java_jar,
            },
            "results": all_results,
            "summary": {
                "total": len(all_results),
                "passed": total_pass,
                "failed": total_fail,
            },
        }
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    main()
