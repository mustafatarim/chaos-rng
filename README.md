# Chaos RNG

[![CI](https://github.com/mustafatarim/chaos-rng/actions/workflows/ci.yml/badge.svg)](https://github.com/mustafatarim/chaos-rng/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Chaos RNG is a Python library for experimenting with random number generation
driven by chaotic three-body dynamics. It provides a `ThreeBodyRNG` API for
direct use, NumPy-compatible generator utilities, and optional validation /
post-processing helpers for offline analysis and integration experiments.

The project is aimed at research, exploration, and reproducible benchmarking of
chaos-based RNG behavior. It is not positioned as a drop-in replacement for
audited cryptographic random number generators, but it includes optional tools
that help evaluate output quality and integration patterns.

This project is currently documented README-first for the `0.1.x` releases.

## Installation

Core package:

```bash
pip install chaos-rng
```

With optional crypto and test utilities:

```bash
pip install "chaos-rng[crypto,test]"
```

## Quick Start

```python
from chaos_rng import ThreeBodyRNG

rng = ThreeBodyRNG(seed=42)

print(rng.random(5))            # floats in [0, 1)
print(rng.randint(0, 10, 5))    # integers
print(rng.bytes(16).hex())      # raw bytes
```

Notes:

- First calls may be slower because numerical kernels warm up (Numba/JIT).
- Use a fixed `seed` for reproducible tests and experiments.

## NumPy Integration

```python
from chaos_rng.generators import create_chaos_generator

gen = create_chaos_generator(seed=42)

normal = gen.normal(size=1000)
uniform = gen.uniform(0, 10, size=100)
integers = gen.integers(0, 100, size=50)
```

## Optional Crypto Utilities

The `crypto` extra provides helper utilities for secure seeding and
post-processing. These tools are useful for experimentation and integration,
but this project should not be treated as a drop-in replacement for a
security-audited cryptographic RNG.

```python
from chaos_rng import ThreeBodyRNG
from chaos_rng.security import CryptoPostProcessor, SecureSeed

seed = SecureSeed().generate_seed()
rng = ThreeBodyRNG(seed=seed)

raw = rng.bytes(64)
processed = CryptoPostProcessor(method="hash").process(raw)

print(len(processed))
```

## Validation Utilities (Optional / Expensive)

Validation helpers are included under `chaos_rng.utils.validation`. They can be
computationally expensive and are best used for offline analysis.

```python
from chaos_rng import ThreeBodyRNG
from chaos_rng.utils.validation import EntropyValidator

rng = ThreeBodyRNG(seed=42)
bits = (rng.random(10000) * 2).astype(int)

validator = EntropyValidator()
result = validator.validate_entropy_rate(bits)
print(result)
```

`NISTTestSuite` and `ContinuousValidator` are also available, but they are not
part of the default CI gate in `0.1.4`.

## Examples

Focused example scripts are available in [`examples/`](examples/README.md):

- `examples/quick_start.py`
- `examples/reproducibility.py`
- `examples/numpy_integration.py`
- `examples/validation_offline.py`
- `examples/crypto_optional.py`

Run one example from the repository root:

```bash
python3 -m pip install -e .[crypto,test]
python3 examples/quick_start.py
```

## Benchmarking

Use the benchmark runner in [`benchmarks/run_benchmarks.py`](benchmarks/run_benchmarks.py)
to collect reproducible JSON results.

Quick benchmark profile:

```bash
python3 -m pip install -e .[crypto,test]
python3 benchmarks/run_benchmarks.py --profile quick --output benchmarks/results/latest.json
```

Fuller benchmark profile:

```bash
python3 benchmarks/run_benchmarks.py --profile full --output benchmarks/results/full.json
```

Benchmark reporting guidance:

- Report cold start (time-to-first-output) separately from warm throughput.
- Include environment details (CPU, OS, Python, NumPy, Numba).
- Compare against baselines (`numpy.random.default_rng`, `random.Random`,
  `os.urandom`, `secrets.token_bytes`) with the same batch sizes.

Suggested README/issue table format:

| Scenario | Mode | Batch | Mean Time (s) | Throughput | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| `ThreeBodyRNG.random` | cold start | 1 | `...` | n/a | includes init + JIT |
| `ThreeBodyRNG.random (lsb)` | warm | 100000 | `...` | `... values/s` | after warm-up |
| `numpy.default_rng().random` | warm | 100000 | `...` | `... values/s` | baseline |
| `ThreeBodyRNG.bytes` | warm | 4096 bytes | `...` | `... bytes/s` | after warm-up |
| `os.urandom` | warm | 4096 bytes | `...` | `... bytes/s` | baseline |

## Performance and Caveats

- This library prioritizes clarity and reproducibility over minimal dependency
  footprint (`numpy`, `scipy`, `numba`).
- Some operations are expensive for large sample sizes.
- Cryptographic helpers are optional and not a substitute for audited crypto
  libraries or formal security review.

## Development

Common commands:

```bash
make install-dev
make lint
make test
make test-slow
make build
make check-build
```

Local release gate:

```bash
make release-check
```

## Release Flow (0.1.x)

The intended release path is:

1. Tag release (`v0.1.4`)
2. Publish to TestPyPI
3. Smoke-test install from TestPyPI
4. Publish to PyPI

GitHub Actions workflows:

- `CI` (`.github/workflows/ci.yml`)
- `Release` (`.github/workflows/release.yml`)

## License

MIT License. See [`LICENSE`](LICENSE).
