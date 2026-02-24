# Benchmarks

Benchmarking in this project is split into two categories:

- `cold_start`: time-to-first-output (includes object creation + JIT warm-up)
- `warm`: steady-state throughput after warm-up

Run a quick profile:

```bash
python3 benchmarks/run_benchmarks.py --profile quick --output benchmarks/results/latest.json
```

Run a fuller profile:

```bash
python3 benchmarks/run_benchmarks.py --profile full --output benchmarks/results/full.json
```

Recommended reporting fields (for README/issues):

- CPU / OS / Python / NumPy / Numba versions
- cold start latency
- warm throughput by batch size
- extraction method comparison (`lsb`, `threshold`, `poincare`)
- baseline comparisons (`numpy.random`, `random`, `os.urandom`, `secrets`)

Important:

- Compare cold and warm results separately.
- Do not present statistical test results as a cryptographic proof.
