# Release Checklist

Practical checklist for publishing `chaos-rng` with minimal ceremony.

## 1) Prepare

- [ ] Update `CHANGELOG.md`
- [ ] Confirm package metadata in `pyproject.toml`
- [ ] Confirm version in:
  - [ ] `pyproject.toml`
  - [ ] `src/chaos_rng/__init__.py`

## 2) Validate locally

```bash
PYTHONPATH=src pytest -q
python -m compileall src
```

- [ ] Tests pass
- [ ] Source compiles without syntax errors

## 3) Build

```bash
python -m build
python -m twine check dist/*
```

- [ ] `sdist` and wheel are generated
- [ ] `twine check` passes

## 4) Publish

Test PyPI first (recommended):

```bash
python -m twine upload --repository testpypi dist/*
```

Production PyPI:

```bash
python -m twine upload dist/*
```

- [ ] Upload succeeded
- [ ] Installation from PyPI works

## 5) Tag and announce

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

- [ ] Tag pushed
- [ ] GitHub release notes created

---

If anything fails, fix and republish with a new version.
