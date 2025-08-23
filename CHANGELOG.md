# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial public release preparation
- Comprehensive documentation
- GitHub Actions CI/CD pipeline
- Pre-commit hooks configuration
- Security scanning and dependency monitoring

## [0.1.0] - 2024-01-XX

### Added
- Three-body gravitational system implementation
- Multiple bit extraction methods (LSB, threshold, Poincaré)
- NumPy-compatible random number generator interface
- NIST SP 800-22 statistical test suite integration
- Cryptographic security features with secure seeding
- Lyapunov exponent calculation for chaos verification
- Thread-safe random number generation
- Performance benchmarking suite
- Comprehensive test coverage
- Documentation with examples and API reference

### Features
- **Core Physics Engine**: Complete three-body problem solver with configurable masses
- **Bit Extraction**: Multiple methods for converting chaotic dynamics to random bits
- **NumPy Integration**: Drop-in replacement for numpy.random with chaos-based entropy
- **Security**: Cryptographically secure seeding and periodic reseeding
- **Validation**: Integrated NIST randomness tests and entropy analysis
- **Performance**: JIT compilation and vectorized operations for high throughput

### Technical Specifications
- Python 3.9+ support
- Cross-platform compatibility (Linux, Windows, macOS)
- Thread-safe design for multi-threaded applications
- Configurable chaos parameters for different entropy sources
- Extensive error handling and input validation

### Documentation
- Complete API documentation
- Usage examples and tutorials
- Mathematical background and references
- Performance benchmarks and statistical validation
- Contributing guidelines and development setup

[Unreleased]: https://github.com/mustafatarim/chaos-rng/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/mustafatarim/chaos-rng/releases/tag/v0.1.0
