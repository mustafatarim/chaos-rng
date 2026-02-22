# Deployment Guide

This guide focuses on the shortest reliable path to publishing `chaos-rng`.

## Prerequisites

- Python 3.9+
- A clean virtual environment
- PyPI account + API token
- (Optional) TestPyPI account + token

## 1. Local verification

```bash
PYTHONPATH=src pytest -q
python -m compileall src
```

If tests are green, continue.

## 2. Build artifacts

```bash
python -m build
python -m twine check dist/*
```

Expected output:
- `dist/chaos_rng-<version>.tar.gz`
- `dist/chaos_rng-<version>-py3-none-any.whl`

## 3. Publish to TestPyPI (recommended)

```bash
python -m twine upload --repository testpypi dist/*
```

Then validate install:

```bash
pip install --index-url https://test.pypi.org/simple/ chaos-rng==<version>
```

## 4. Publish to PyPI

```bash
python -m twine upload dist/*
```

## 5. Post-release smoke test

```bash
pip install chaos-rng==<version>
python -c "from chaos_rng import ThreeBodyRNG; print(ThreeBodyRNG(seed=42).random())"
```

## 6. Tag release

```bash
git tag v<version>
git push origin v<version>
```

## Notes

- Keep claims conservative: this is a chaos-based RNG library, not a certified crypto module.
- For cryptographic keys/tokens in production apps, prefer Python `secrets` or audited system RNGs.
