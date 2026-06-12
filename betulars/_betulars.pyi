"""Type stubs for betulars - Python bindings for BETULA clustering."""
from __future__ import annotations

import numpy as np

__version__: str

class Betula:
    """High-level BETULA clustering wrapper.
    
    Builds a CF-Tree from data in one call.
    
    Args:
        data: 2D numpy array of shape (n_points, n_dims), dtype float64.
        capacity: Node capacity (default 32).
        maxleaves: Maximum leaf entries before rebuild (default 1000).
        threshold: Leaf entry size threshold (default 0.0).
        distance: Distance measure for tree routing (default "euclidean").
        absorption: Distance measure for leaf absorption (default "euclidean").
        feature: Cluster feature type — "vii", "vvi", or "vvv" (default "vii").
    """
    def __init__(
        self,
        data: np.ndarray,
        capacity: int = 32,
        maxleaves: int = 1000,
        threshold: float = 0.0,
        distance: str = "euclidean",
        absorption: str = "euclidean",
        feature: str = "vii",
    ) -> None: ...
    @property
    def leaf_clusters(self) -> list[ClusterFeature]: ...
    @property
    def num_clusters(self) -> int: ...
    @property
    def rebuild_count(self) -> int: ...
    @property
    def overall_variance(self) -> float: ...

class CFTree:
    """Incremental CF-Tree for streamable clustering.
    
    Insert points one at a time and access the resulting leaf clusters.
    
    Args:
        dim: Dimensionality of the data points.
        capacity: Node capacity (default 32).
        maxleaves: Maximum leaf entries before rebuild (default 1000).
        threshold: Leaf entry size threshold (default 0.0).
        feature: Cluster feature type — "vii", "vvi", or "vvv" (default "vii").
    """
    def __init__(
        self,
        dim: int,
        capacity: int = 32,
        maxleaves: int = 1000,
        threshold: float = 0.0,
        feature: str = "vii",
    ) -> None: ...
    def insert(self, point: np.ndarray | list[float]) -> None:
        """Insert a single data point into the tree.
        
        Args:
            point: 1D numpy array or sequence of floats.
        """
    @property
    def leaf_clusters(self) -> list[ClusterFeature]: ...
    @property
    def num_clusters(self) -> int: ...
    @property
    def rebuild_count(self) -> int: ...
    @property
    def dim(self) -> int: ...

class ClusterFeature:
    """Read-only cluster statistics.
    
    Attributes:
        size: Number of points in the cluster.
        centroid: Cluster centroid as a numpy 1D array.
        ssd: Sum of squared deviations.
        ssd_per_dim: Per-dimension SSDs as a numpy 1D array.
        covariance: Full covariance matrix (numpy 2D array), or None if unavailable.
        variance: ssd / size, or 0.0 if empty.
    """
    @property
    def size(self) -> int: ...
    @property
    def centroid(self) -> np.ndarray: ...
    @property
    def ssd(self) -> float: ...
    @property
    def ssd_per_dim(self) -> np.ndarray: ...
    @property
    def covariance(self) -> np.ndarray | None: ...
    @property
    def variance(self) -> float: ...