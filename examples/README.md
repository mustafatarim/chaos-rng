# Examples

This directory contains small, focused examples that are easier to run and
reason about than the all-in-one `demo.py`.

Recommended order:

1. `quick_start.py`
2. `reproducibility.py`
3. `numpy_integration.py`
4. `validation_offline.py`
5. `crypto_optional.py`

Run examples from the repository root after installing the package in editable
mode:

```bash
python3 -m pip install -e .[crypto,test]
python3 examples/quick_start.py
```
