# BETULA

Fast, streamable data clustering in Rust — an improved implementation of the BIRCH algorithm with better cluster features.

## What is BETULA?

BETULA introduces improved [Clustering Feature (CF) trees](https://en.wikipedia.org/wiki/CF_tree) that avoid the catastrophic cancellation present in the original BIRCH algorithm. The key contribution is a replacement cluster feature that:

- Is numerically stable (no catastrophic cancellation)
- Is not much more expensive to maintain
- Makes many computations simpler and more efficient
- Can be used in other BIRCH-derived algorithms (streaming, k-means, GMM, hierarchical clustering)

This crate is a Rust implementation based on [ELKI's BETULA](https://github.com/elki-project/elki) for the Java reference.

> **Reference:** Lang & Schubert, *"BETULA: Fast clustering of large data with improved BIRCH CF-Trees"*, Information Systems 108 (2022), [DOI: 10.1016/J.IS.2021.101918](https://doi.org/10.1016/J.IS.2021.101918)

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
betula = "0.1"
```

## Quick Start

```rust
use betula::cf_tree::CFTree;
use betula::cluster_feature::VII;
use betula::distance::CentroidEuclideanDistance;

// Create a CF-Tree with Euclidean distance
let mut tree = CFTree::new(
    CentroidEuclideanDistance::new(),  // distance function
    CentroidEuclideanDistance::new(),  // absorption function
    32,       // node capacity
    2,        // dimensionality
    1000,     // max leaf entries
    0.0,      // threshold (auto-estimated)
);

// Insert points
for point in &[
    vec![1.0, 2.0],
    vec![1.1, 2.1],
    vec![10.0, 20.0],
    vec![10.1, 20.1],
] {
    tree.insert(point);
}

// Access cluster statistics
for (id, cf) in tree.leaf_entries().iter().enumerate() {
    println!(
        "Cluster {}: {} points, centroid={:?}, variance={:.4}",
        id,
        cf.size(),
        cf.centroid(),
        cf.ssd() / cf.size() as f64,
    );
}
```

## CF-Tree

The core data structure is a `CFTree<F, CF, D, A>` parameterized by:

| Parameter | Description |
|-----------|-------------|
| `F` | Float type (e.g. `f64`) |
| `CF` | Cluster feature type — `VII`, `VVI`, or `VVV` |
| `D` | Distance function for tree routing |
| `A` | Absorption function for leaf assignment |

### Cluster Features

| Feature | Stores | Use case |
|---------|--------|----------|
| `VII` | Scalar SSD + centroid | Simple clustering, most common |
| `VVI` | Per-dimension SSD + centroid | When per-dimension variance matters |
| `VVV` | Full cross-product matrix + centroid | When covariance is needed |

### Parameters

- **capacity**: Node capacity — controls tree depth. Lower values give deeper trees; higher values give shallower trees. Recommended: 32–64.
- **maxleaves**: Maximum leaf entries — triggers tree rebuilds when exceeded.
- **threshold**: Leaf entry size limit — automatically estimated on rebuild. A good threshold avoids rebuilds; starting with `0` is fine.

> **Note:** The Python bindings (`betulars`) provide default values for these parameters (`capacity=32`, `maxleaves=1000`, `threshold=0.0`). The Rust API requires all parameters explicitly.

### Insertion API

- `insert(x)` — validated insertion (checks dimensionality)
- `insert_unchecked(x)` — unsafe fast path (caller guarantees `x.len() >= dim`)

## Distance Measures

Six distance functions are available, all implementing the `CFDistance` trait:

| Distance | Description |
|----------|-------------|
| `CentroidEuclideanDistance` | Squared Euclidean distance between centroids |
| `CentroidManhattanDistance` | Manhattan distance between centroids |
| `AverageInterclusterDistance` | Average squared distance to cluster points (Zhang "D2") |
| `AverageIntraclusterDistance` | Average pairwise distance within clusters (Lang & Schubert "D3") |
| `VarianceIncreaseDistance` | SSD increase from adding/merging (Zhang "D4") |
| `RadiusDistance` | Average radius of the resulting cluster |

## CLI Benchmark Tool

A benchmark binary is included for performance testing and comparison:

```bash
cargo build --release

./target/release/betula \
  --input data.npy \
  --output results.json \
  --capacity 32 \
  --maxleaves 1000 \
  --distance euclidean
```

**Output:** JSON with metadata, timing, per-cluster statistics (centroid, size, variance).

## Python Bindings

BETULA can be used from Python via the **betulars** package, built with [pyo3](https://github.com/PyO3/pyo3) and [maturin](https://github.com/PyO3/maturin).

### Installation

From source:

```bash
pip install maturin
maturin develop --features python
```

Or build a wheel:

```bash
maturin build --features python
pip install target/wheels/betulars-*.whl
```

### Quick Start

```python
import numpy as np
import betulars

# High-level: build a model from data
data = np.array([[1.0, 2.0], [1.1, 2.1], [10.0, 20.0], [10.1, 20.1]])
model = betulars.Betula(
    data=data,
    capacity=32,
    maxleaves=1000,
    threshold=0.0,
    distance="euclidean",
    absorption="euclidean",
    feature="vii",  # or "vvi" or "vvv"
)

print(f"Clusters: {model.num_clusters}")
print(f"Variance: {model.overall_variance:.4f}")

for cf in model.leaf_clusters:
    print(f"  Cluster: {cf.size} points, centroid={cf.centroid}, variance={cf.variance:.4f}")
```

### Incremental Insertion

```python
import numpy as np
import betulars

# Lower-level: insert points one at a time
tree = betulars.CFTree(dim=2, capacity=32, maxleaves=1000, threshold=0.0, feature="vii")

for point in np.array([[1.0, 2.0], [1.1, 2.1], [10.0, 20.0]]):
    tree.insert(point)

print(f"Clusters: {tree.num_clusters}")
for cf in tree.leaf_clusters:
    print(f"  {cf.size} points, centroid={cf.centroid}")
```

### Supported Distance Measures

| Name | Description |
|------|-------------|
| `euclidean` | Squared Euclidean distance between centroids |
| `manhattan` | Manhattan distance between centroids |
| `avgintercluster` | Average squared distance to cluster points |
| `avgintracluster` | Average pairwise distance within clusters |
| `varianceincrease` | SSD increase from adding/merging |
| `radius` | Average radius of the resulting cluster |

### Cluster Feature Types

| Feature | Stores | Use case |
|---------|--------|----------|
| `vii` | Scalar SSD + centroid | Simple clustering, most common |
| `vvi` | Per-dimension SSD + centroid | When per-dimension variance matters |
| `vvv` | Full cross-product matrix + centroid | When covariance is needed |

### API Reference

#### `Betula`

High-level wrapper. Builds a CF-Tree from data in one call.

- `Betula(data, capacity=32, maxleaves=1000, threshold=0.0, distance="euclidean", absorption="euclidean", feature="vii")`
  - `data` — 2D numpy array, shape `(n_points, n_dims)`, dtype `float64`
- `.leaf_clusters` — list of `ClusterFeature` objects
- `.num_clusters` — number of leaf clusters
- `.rebuild_count` — number of tree rebuilds
- `.overall_variance` — total SSD / total points

#### `CFTree`

Incremental CF-Tree. Insert points one at a time.

- `CFTree(dim, capacity=32, maxleaves=1000, threshold=0.0, feature="vii")`
- `.insert(point)` — insert a data point (1D numpy array or sequence)
- `.leaf_clusters` — list of `ClusterFeature` objects
- `.num_clusters` — number of leaf clusters
- `.rebuild_count` — number of tree rebuilds
- `.dim` — dimensionality

#### `ClusterFeature`

Read-only cluster statistics.

- `.size` — number of points
- `.centroid` — numpy 1D array
- `.ssd` — sum of squared deviations
- `.ssd_per_dim` — numpy 1D array of per-dimension SSDs
- `.covariance` — numpy 2D array (or None for vii/vvi)
- `.variance` — ssd / size

### Python Version Support

Python 3.8 and later (via abi3 compatibility).

## Roadmap

The following features are planned for future releases:

- **K-means clustering** — Refine CF-Tree leaf clusters with k-means for a fast k-means approximation.
- **Gaussian Mixture Modelling** — Fit GMMs to leaf clusters using the full covariance information from `VVV` cluster features.
- **Parallel tree building** — Multi-threaded tree construction for improved performance on large datasets.
- **Hierarchical agglomerative clustering** — Bottom-up merging of leaf clusters to produce a full hierarchy / dendrogram.

## License

Dual licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE).
