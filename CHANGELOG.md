# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.3] - 2026-02-24

### Added

- Focused example scripts under `examples/` for quick-start, reproducibility,
  NumPy integration, offline validation, and optional crypto helpers.
- Reproducible benchmark runner under `benchmarks/` with JSON output, baseline
  comparisons, and separate cold-start vs warm-throughput reporting.

### Changed

- Expanded README with example/benchmark usage guidance and a clearer project
  positioning intro.
- Included `examples/` and `benchmarks/` in source distribution packaging.

## [0.1.2] - 2026-02-22

### Fixed

- Stopped emitting a `nistrng` warning during normal `chaos_rng` import when the
  optional NIST dependency is not installed.
- NIST helpers still raise an explicit `ImportError` when `NISTTestSuite` is used
  without installing `nistrng`.

## [0.1.1] - 2026-02-22

### Changed

- Simplified CI to a lean `ci.yml` pipeline and a focused release workflow.
- Removed broken docs/Read the Docs automation in favor of a README-first flow.
- Simplified `pre-commit` hooks and local developer commands in `Makefile`.
- Updated package metadata and release tooling for `0.1.1`.
- Reduced default QA scope to reliable, non-empty test/lint/build checks.

## [0.1.0] - 2024-01-XX

### Added

- Three-body gravitational system implementation
- Multiple bit extraction methods (LSB, threshold, Poincaré)
- NumPy-compatible random number generator interface
- Validation utilities and statistical test helpers
- Optional cryptographic utilities for seeding and post-processing
- Performance benchmarking and test coverage

[Unreleased]: https://github.com/mustafatarim/chaos-rng/compare/v0.1.3...HEAD
[0.1.3]: https://github.com/mustafatarim/chaos-rng/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/mustafatarim/chaos-rng/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/mustafatarim/chaos-rng/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/mustafatarim/chaos-rng/releases/tag/v0.1.0
