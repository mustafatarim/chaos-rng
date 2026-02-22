# chaos-rng

Minimal, deterministic and NumPy-friendly random number generation based on chaotic three-body dynamics.

> ⚠️ `chaos-rng` is a research/engineering RNG. It is **not** a replacement for audited cryptographic RNGs such as `secrets`.

## Why this project?

This library keeps one core idea: use a chaotic physical simulation as an entropy source, then expose a familiar Python API.

Main goals:
- Keep the core API small (`random`, `randint`, `bytes`).
- Keep output reproducible when a seed is provided.
- Keep NumPy integration straightforward.
- Keep docs and release process practical.

## Installation

```bash
pip install chaos-rng
```

For development:

```bash
pip install -e .
```

## Quick start

```python
from chaos_rng import ThreeBodyRNG

rng = ThreeBodyRNG(seed=42)

print(rng.random())          # float in [0, 1)
print(rng.random(5))         # numpy array
print(rng.randint(0, 10, 4)) # integers in [0, 10)
print(rng.bytes(16).hex())   # random bytes
```

## NumPy-compatible usage

```python
from chaos_rng.generators import create_chaos_generator

g = create_chaos_generator(seed=42)

samples = g.normal(loc=0, scale=1, size=1000)
```

## API overview

### `ThreeBodyRNG`
- `random(size=None)` → float/ndarray in `[0, 1)`
- `randint(low, high, size=None)` → int/ndarray in `[low, high)`
- `bytes(length)` → `bytes`
- `get_state()` / `set_state(...)` → state snapshot for reproducible sequences

### `ThreeBodySystem`
- `evolve(dt, steps)`
- `compute_lyapunov_exponent(...)`
- `get_energy()`

### NumPy layer
- `ChaosBitGenerator`
- `ChaosGenerator`
- `create_chaos_generator(seed=...)`

## Project structure

```text
src/chaos_rng/
  generators/     # RNG logic and NumPy compatibility
  integrators/    # ODE solver and Lyapunov tools
  utils/          # validation and analysis helpers
tests/            # unit tests
```

## Development checks

Because this repository uses a `src/` layout, run tests with `PYTHONPATH=src` if you are not installing the package first:

```bash
PYTHONPATH=src pytest -q
```

## Release (simple workflow)

1. Update `CHANGELOG.md`.
2. Bump version in:
   - `pyproject.toml`
   - `src/chaos_rng/__init__.py`
3. Run checks:

```bash
PYTHONPATH=src pytest -q
python -m compileall src
```

4. Build package:

```bash
python -m build
```

5. Upload:

```bash
twine upload dist/*
```

## License

MIT — see [LICENSE](LICENSE).
