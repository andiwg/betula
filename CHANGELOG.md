# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Three cluster feature types: `VII` (scalar SSD), `VVI` (per-dimension SSD), `VVV` (full cross-product matrix)
- Six distance measures: `CentroidEuclideanDistance`, `CentroidManhattanDistance`, `AverageInterclusterDistance`, `AverageIntraclusterDistance`, `VarianceIncreaseDistance`, `RadiusDistance`
- 8-lane unrolled `SqEuclidean` and `Manhattan` vector distance implementations (FMA)
- CFTree rebuild with automatic threshold estimation
- Benchmark CLI tool (`betula`) with JSON output
- Regression tests with golden values for multiple distance measures
- Python comparison script for Rust vs Java (ELKI) verification

### Changed
- Refactored CFTree to use slice-based access API

### Removed
- `gen_data` binary (replaced by numpy-based data generation in comparison script)

### Fixed
- CF double-update bug during split propagation
- Split tie-breaking to balance groups by entry count
