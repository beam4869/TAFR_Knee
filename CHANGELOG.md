# Changelog

All notable changes to this project are documented here. The project follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-09-04

### Added

- Initial `tafr-knee` package and `tafrknee` public namespace.
- `select_knee`, `discover_knees`, and `audit_weight` APIs.
- Tabular, SciPy callable, generic callback, and optional Pyomo problem adapters.
- Frozen payoff-matrix or user-specified objective normalization.
- Complete clipped perturbation-polytope vertex enumeration for any objective count.
- Pairwise, vertex, hybrid, and adaptive-adversarial finite-radius audit strategies.
- Non-extreme anchor screening, plateau-aware stability estimation, trade-off-active
  exit screening, and `K_exit` certification.
- Original-space certification after linear objective reduction.
- Deterministic simplex candidate generation with Sobol fallback, solve caching,
  warm starts, and optional threaded evaluation.
- JSON CLI for objective tables, plotting helpers, examples, tests, and CI.

[Unreleased]: https://github.com/beam4869/TAFR_Knee/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/beam4869/TAFR_Knee/releases/tag/v0.1.0
