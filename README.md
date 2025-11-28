# Chaos RNG: Professional Chaos-Based Random Number Generator

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/mustafatarim/chaos-rng/workflows/Tests/badge.svg)](https://github.com/mustafatarim/chaos-rng/actions)

A professional-grade Python library for generating high-quality random numbers using the chaotic dynamics of three-body gravitational systems. This library combines advanced numerical physics with cryptographic security features to provide a robust entropy source suitable for both research and practical applications.

## 🌟 Features

### Core Capabilities
- **Three-Body Physics**: Implements the gravitational three-body problem with customizable masses
- **Multiple Bit Extraction Methods**: LSB, threshold-based, and Poincaré section sampling
- **NumPy Integration**: Full compatibility with NumPy's random number generation framework
- **Cryptographic Security**: Secure seeding, periodic reseeding, and post-processing options
- **Statistical Validation**: Integrated NIST SP 800-22 test suite and entropy analysis

### Performance & Quality
- **High Performance**: Numba JIT compilation and vectorized operations
- **Excellent Statistics**: Passes all NIST statistical tests for randomness
- **Chaotic Verification**: Lyapunov exponent calculation confirms chaotic behavior
- **Thread Safety**: Per-thread instances for multi-threaded applications
- **Forward Secrecy**: Automatic reseeding prevents state recovery attacks

## 🚀 Quick Start

### Installation

```bash
pip install chaos-rng
```

For all features including cryptographic enhancements:
```bash
pip install chaos-rng[crypto,test]
```

### Basic Usage

```python
import numpy as np
from chaos_rng import ThreeBodyRNG

# Create a chaos-based random number generator
rng = ThreeBodyRNG(seed=42)

# Generate random floats in [0, 1)
random_floats = rng.random(10)
print(f"Random floats: {random_floats}")

# Generate random integers
random_ints = rng.randint(0, 100, size=5)
print(f"Random integers: {random_ints}")

# Generate cryptographic random bytes
random_bytes = rng.bytes(32)
print(f"Random bytes: {random_bytes.hex()}")
```

### NumPy Integration

```python
from chaos_rng.generators import create_chaos_generator
import numpy as np

# Create NumPy-compatible generator
rng = create_chaos_generator(seed=42)

# Use with all NumPy distributions
normal_samples = rng.normal(loc=0, scale=1, size=1000)
uniform_samples = rng.uniform(0, 10, size=500)
exponential_samples = rng.exponential(scale=2.0, size=750)

# Advanced distributions
gamma_samples = rng.gamma(shape=2, scale=1, size=1000)
beta_samples = rng.beta(a=2, b=5, size=500)
```

## 🔬 Advanced Usage

### Custom Three-Body Systems

```python
from chaos_rng import ThreeBodyRNG

# Custom masses for different chaotic dynamics
custom_masses = [2.0, 1.5, 0.8]
rng = ThreeBodyRNG(
    seed=42,
    masses=custom_masses,
    extraction_method='threshold'
)

# Verify chaotic behavior
lyapunov_exponent = rng.system.compute_lyapunov_exponent()
print(f"Lyapunov exponent: {lyapunov_exponent}")

if lyapunov_exponent > 0:
    print("System exhibits chaotic behavior ✓")
```

### Bit Extraction Methods

```python
# Compare different extraction methods
methods = ['lsb', 'threshold', 'poincare', 'combined']

for method in methods:
    rng = ThreeBodyRNG(seed=42, extraction_method=method)
    samples = rng.random(10000)

    # Basic statistical check
    mean = np.mean(samples)
    std = np.std(samples)
    print(f"{method:10s}: mean={mean:.4f}, std={std:.4f}")
```

### Statistical Validation

```python
from chaos_rng.utils.validation import NISTTestSuite, EntropyValidator

# Generate test sequence
rng = ThreeBodyRNG(seed=42)
binary_sequence = (rng.random(100000) * 2).astype(int)

# Run NIST statistical tests
nist_suite = NISTTestSuite(significance_level=0.01)
nist_results = nist_suite.run_all_tests(binary_sequence)

# Check results
passed_tests = sum(1 for result in nist_results.values() if result.passed)
total_tests = len(nist_results)
print(f"NIST Tests: {passed_tests}/{total_tests} passed")

# Entropy validation
entropy_validator = EntropyValidator()
entropy_results = entropy_validator.validate_entropy_rate(binary_sequence)
print(f"Entropy validation: {entropy_results['overall_valid']}")
```

### Cryptographic Applications

```python
from chaos_rng.security import CryptoPostProcessor

# Create RNG with cryptographic post-processing
rng = ThreeBodyRNG(seed=42)
crypto_processor = CryptoPostProcessor(method='chacha20')

# Generate cryptographically secure random data
raw_data = rng.bytes(1000)
secure_data = crypto_processor.process(raw_data)

print(f"Generated {len(secure_data)} cryptographically secure bytes")
```

### Continuous Monitoring

```python
from chaos_rng.utils.validation import ContinuousValidator

# Set up continuous validation
validator = ContinuousValidator(
    window_size=50000,
    test_interval=10000,
    alert_threshold=3
)

rng = ThreeBodyRNG(seed=42)

# Generate data and monitor quality
for i in range(10):
    new_data = (rng.random(10000) * 2).astype(int)
    validation_results = validator.add_data(new_data)

    if validation_results:
        pass_rate = validation_results['summary']['pass_rate']
        print(f"Batch {i+1}: Quality = {pass_rate:.2%}")

        if validator.failure_count >= 3:
            print("⚠️  Quality alert: Consider reseeding")

# Get overall status
status = validator.get_status()
print(f"Overall pass rate: {status['overall_pass_rate']:.2%}")
```

## 🔧 Configuration

### System Parameters

```python
# Fine-tune the three-body system
rng = ThreeBodyRNG(
    seed=42,
    masses=[1.2, 0.8, 1.5],           # Body masses
    extraction_method='combined',      # Bit extraction method
    buffer_size=20000,                 # Internal buffer size
    reseed_interval=2000000           # Automatic reseeding interval
)

# Access system properties
print(f"System energy: {rng.system.get_energy()}")
print(f"Current time: {rng.system.time}")
```

### Performance Optimization

```python
# For high-performance applications
from chaos_rng.generators import ThreadSafeGenerator

# Thread-safe generator for concurrent use
thread_safe_rng = ThreadSafeGenerator(
    seed=42,
    masses=[1.0, 1.0, 1.0],
    extraction_method='lsb'  # Fastest method
)

# Benchmark performance
benchmark_results = thread_safe_rng.benchmark_performance(n_samples=1000000)
print(f"Generation rate: {benchmark_results['random_rate']:.0f} samples/sec")
```

## 🧪 Scientific Background

### Three-Body Problem

The three-body problem describes the motion of three gravitationally interacting point masses:

```
m_i * d²r_i/dt² = Σ(j≠i) G*m_i*m_j*(r_j - r_i)/|r_j - r_i|³
```

This system exhibits chaotic behavior, meaning small changes in initial conditions lead to exponentially diverging trajectories. The chaos provides an excellent source of entropy for random number generation.

### Lyapunov Exponents

The largest Lyapunov exponent λ quantifies the rate of divergence:

```
λ = lim(t→∞) (1/t) * ln(|δ(t)|/|δ₀|)
```

A positive λ confirms chaotic behavior. Our implementation typically produces λ > 0.1, indicating strong chaos.

### Bit Extraction Methods

1. **LSB (Least Significant Bit)**: `bit = int(abs(coord) * 2³²) % 2`
2. **Threshold**: `bit = 1 if x(n+1) > x(n) else 0`
3. **Poincaré Section**: Samples bits when trajectories cross hyperplanes

## 🔐 Security Considerations

### Cryptographic Strength

- **Entropy Source**: High-dimensional chaotic dynamics provide excellent entropy
- **Secure Seeding**: Uses `os.urandom()` and multiple entropy sources
- **Forward Secrecy**: Automatic reseeding prevents state recovery
- **Side-Channel Resistance**: Constant-time operations where applicable

### Security Recommendations

1. **Production Use**: Enable cryptographic post-processing
2. **Key Generation**: Use the `bytes()` method for cryptographic keys
3. **Regular Validation**: Monitor output quality with continuous validation
4. **Proper Seeding**: Use high-entropy seeds in security applications

```python
# Recommended configuration for cryptographic use
from chaos_rng.security import SecureSeed, CryptoPostProcessor

secure_seed = SecureSeed().generate_seed()
crypto_processor = CryptoPostProcessor(method='chacha20')

rng = ThreeBodyRNG(seed=secure_seed)
```

## 📊 Performance Benchmarks

Typical performance on modern hardware:

| Operation | Rate (samples/sec) | Notes |
|-----------|-------------------|--------|
| `random()` | ~500K | Floating-point generation |
| `randint()` | ~400K | Integer generation |
| `bytes()` | ~50MB/s | Raw byte generation |
| NumPy distributions | ~300K | Via NumPy compatibility |

Performance scales well with CPU cores when using thread-safe generators.

## 🧑‍💻 Development

### Running Tests

```bash
# Install development dependencies
pip install -e .[dev,test]

# Run all tests
pytest

# Run with coverage
pytest --cov=chaos_rng --cov-report=html

# Run statistical tests (slow)
pytest -m statistical

# Skip slow tests
pytest -m "not slow"
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/
```

## 📚 API Reference

### Core Classes

- **`ThreeBodyRNG`**: Main random number generator class
- **`ChaosBitGenerator`**: NumPy-compatible BitGenerator
- **`ChaosGenerator`**: NumPy-compatible Generator with all distributions

### Validation Tools

- **`NISTTestSuite`**: NIST SP 800-22 statistical tests
- **`EntropyValidator`**: Entropy measurement and validation
- **`ContinuousValidator`**: Real-time quality monitoring

### Security Utilities

- **`SecureSeed`**: Secure seed generation
- **`CryptoPostProcessor`**: Cryptographic post-processing
- **`EntropyMixer`**: Multiple entropy source mixing

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Areas for Contribution

- Additional chaotic systems (Lorenz, Rössler, etc.)
- GPU acceleration with CuPy
- Hardware RNG integration
- Additional bit extraction methods
- Performance optimizations

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Based on numerical methods from SciPy
- Statistical tests from NIST SP 800-22
- Performance optimization with Numba
- Cryptographic functions from the `cryptography` library

## 📖 Citation

If you use this library in academic research, please cite:

```bibtex
@software{chaos_rng,
  title={Chaos RNG: Professional Chaos-Based Random Number Generator},
  author={Mustafa Tarim},
  year={2025},
  url={https://github.com/mustafatarim/chaos-rng}
}
```

## 🔗 Related Projects

- [NumPy](https://numpy.org/) - Numerical computing library
- [SciPy](https://scipy.org/) - Scientific computing library
- [NIST Test Suite](https://csrc.nist.gov/Projects/Random-Bit-Generation) - Statistical tests for RNGs
- [Cryptography](https://cryptography.io/) - Cryptographic recipes and primitives

---

**Note**: This library is suitable for research and educational purposes. For production cryptographic applications, ensure proper security review and validation.
