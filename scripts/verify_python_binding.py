#!/usr/bin/env python3
"""
Compare betulars (Python) output against the native Rust betula binary.

Generates deterministic test data, runs both implementations with matching
parameters, and verifies that cluster counts, sizes, centroids, and overall
variance agree within numerical tolerance.

Usage:
    python3 scripts/verify_python_binding.py
    python3 scripts/verify_python_binding.py --rust-bin /path/to/betula
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# ── defaults ──────────────────────────────────────────────────────────

DEFAULT_RUST_BIN = PROJECT_ROOT / "target/release/betula"
TOLERANCE = 1e-6  # centroid / variance comparison tolerance

# ── parameter grid ────────────────────────────────────────────────────

PARAMS = [
    {
        "capacity": 8,
        "maxleaves": 50,
        "threshold": 0.0,
        "distance": "euclidean",
        "absorption": "euclidean",
    },
    {
        "capacity": 32,
        "maxleaves": 1000,
        "threshold": 0.0,
        "distance": "euclidean",
        "absorption": "euclidean",
    },
    {
        "capacity": 64,
        "maxleaves": 200,
        "threshold": 0.5,
        "distance": "manhattan",
        "absorption": "manhattan",
    },
    {
        "capacity": 16,
        "maxleaves": 100,
        "threshold": 0.0,
        "distance": "euclidean",
        "absorption": "varianceincrease",
    },
    {
        "capacity": 32,
        "maxleaves": 500,
        "threshold": 0.0,
        "distance": "avgintercluster",
        "absorption": "radius",
    },
]

# ── helpers ───────────────────────────────────────────────────────────


def generate_data(seed: int, n_points: int, dims: int) -> np.ndarray:
    """Generate deterministic clustered data."""
    rng = np.random.RandomState(seed)
    n_clusters = max(3, dims)
    centroids = rng.randn(n_clusters, dims) * 10
    data = []
    for c in centroids:
        cluster_size = n_points // n_clusters
        data.append(rng.randn(cluster_size, dims) * 0.5 + c)
    data = np.vstack(data).astype(np.float64)
    rng.shuffle(data)
    return data


def run_rust(data_path: Path, output_path: Path, rust_bin: Path, params: dict) -> dict:
    """Run the Rust betula binary and return parsed JSON results."""
    cmd = [
        str(rust_bin),
        "--input",
        str(data_path),
        "--output",
        str(output_path),
        "--capacity",
        str(params["capacity"]),
        "--maxleaves",
        str(params["maxleaves"]),
        "--threshold",
        str(params["threshold"]),
        "--distance",
        params["distance"],
        "--absorption",
        params["absorption"],
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        print(f"  Rust binary failed:\n{result.stderr}")
        sys.exit(1)
    with open(output_path) as f:
        return json.load(f)


def run_python(data: np.ndarray, params: dict) -> dict:
    """Run betulars and return comparable results."""
    import betulars

    model = betulars.Betula(
        data=data,
        capacity=params["capacity"],
        maxleaves=params["maxleaves"],
        threshold=params["threshold"],
        distance=params["distance"],
        absorption=params["absorption"],
    )

    clusters = []
    for cf in model.leaf_clusters:
        clusters.append(
            {
                "size": cf.size,
                "centroid": cf.centroid,
                "variance": cf.variance,
            }
        )

    return {
        "num_clusters": model.num_clusters,
        "rebuild_count": model.rebuild_count,
        "overall_variance": model.overall_variance,
        "clusters": clusters,
    }


def compare(rust_results: dict, python_results: dict, params: dict) -> list[str]:
    """Compare Rust and Python results. Return list of mismatch descriptions."""
    errors: list[str] = []

    # Cluster count
    rust_clusters = len(rust_results["clusters"])
    py_clusters = python_results["num_clusters"]
    if rust_clusters != py_clusters:
        errors.append(
            f"  Cluster count mismatch: rust={rust_clusters}, python={py_clusters}"
        )

    # Overall variance
    rust_var = rust_results["overall_variance"]
    py_var = python_results["overall_variance"]
    if abs(rust_var - py_var) > TOLERANCE:
        errors.append(
            f"  Overall variance mismatch: rust={rust_var:.8f}, python={py_var:.8f}"
        )

    # Per-cluster: sizes must match, centroids within tolerance
    # Rust clusters are in leaf-entry order; Python is too.
    # Sort both by centroid[0] to align them.
    rust_sorted = sorted(rust_results["clusters"], key=lambda c: c["centroid"][0])
    py_sorted = sorted(python_results["clusters"], key=lambda c: c["centroid"][0])

    for i, (rc, pc) in enumerate(zip(rust_sorted, py_sorted)):
        # Size
        if rc["size"] != pc["size"]:
            errors.append(f"  Cluster {i} size: rust={rc['size']}, python={pc['size']}")

        # Centroid
        rc_cent = np.array(rc["centroid"])
        pc_cent = np.array(pc["centroid"])
        if np.any(np.abs(rc_cent - pc_cent) > TOLERANCE):
            max_diff = np.max(np.abs(rc_cent - pc_cent))
            errors.append(
                f"  Cluster {i} centroid max diff={max_diff:.2e}: "
                f"rust={rc_cent[:3].tolist()}{('...' if len(rc_cent) > 3 else '')}, "
                f"py={pc_cent[:3].tolist()}{('...' if len(pc_cent) > 3 else '')}"
            )

        # Variance
        if abs(rc["variance"] - pc["variance"]) > TOLERANCE:
            errors.append(
                f"  Cluster {i} variance: rust={rc['variance']:.8f}, python={pc['variance']:.8f}"
            )

    return errors


def main():
    parser = argparse.ArgumentParser(
        description="Compare betulars (Python) against native Rust betula."
    )
    parser.add_argument(
        "--rust-bin",
        type=Path,
        default=DEFAULT_RUST_BIN,
        help="Path to the Rust betula binary (default: target/release/betula)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for data generation"
    )
    parser.add_argument(
        "--points", type=int, default=50_000, help="Number of data points"
    )
    parser.add_argument("--dims", type=int, default=3, help="Number of dimensions")
    args = parser.parse_args()

    if not args.rust_bin.exists():
        print(f"Rust binary not found: {args.rust_bin}")
        print("Build it first: cargo build --release")
        sys.exit(1)

    # Build test data
    data = generate_data(args.seed, args.points, args.dims)
    print(f"Data: {data.shape[0]} points × {data.shape[1]} dims (seed={args.seed})")
    print()

    all_ok = True
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        data_path = tmpdir / "test.npy"
        np.save(data_path, data)
        output_path = tmpdir / "results.json"

        for i, params in enumerate(PARAMS):
            label = (
                f"cap={params['capacity']} "
                f"maxleaves={params['maxleaves']} "
                f"thresh={params['threshold']} "
                f"d={params['distance']} "
                f"a={params['absorption']}"
            )
            print(f"[{i + 1}/{len(PARAMS)}] {label}")

            # Rust
            rust_results = run_rust(data_path, output_path, args.rust_bin, params)

            # Python
            python_results = run_python(data, params)

            # Compare
            errors = compare(rust_results, python_results, params)
            if errors:
                all_ok = False
                for e in errors:
                    print(e)
            else:
                print(
                    f"  ✓ OK (clusters={python_results['num_clusters']}, "
                    f"variance={python_results['overall_variance']:.6f})"
                )
            print()

    if all_ok:
        print(f"All {len(PARAMS)} parameter sets matched.")
    else:
        print("Some parameter sets had mismatches.")
        sys.exit(1)


if __name__ == "__main__":
    main()
