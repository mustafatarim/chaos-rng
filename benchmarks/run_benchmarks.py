#!/usr/bin/env python3
"""Reproducible benchmark runner for chaos-rng.

This script separates cold-start latency (including JIT warm-up) from warm
throughput measurements and stores results as JSON for later comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import secrets
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from chaos_rng import ThreeBodyRNG
from chaos_rng.generators import create_chaos_generator


@dataclass
class TimingStats:
    runs: int
    min_seconds: float
    max_seconds: float
    mean_seconds: float
    median_seconds: float
    stdev_seconds: float

    @classmethod
    def from_samples(cls, samples: list[float]) -> TimingStats:
        if not samples:
            raise ValueError("timing sample list cannot be empty")
        return cls(
            runs=len(samples),
            min_seconds=min(samples),
            max_seconds=max(samples),
            mean_seconds=statistics.mean(samples),
            median_seconds=statistics.median(samples),
            stdev_seconds=statistics.stdev(samples) if len(samples) > 1 else 0.0,
        )


def bench_timed(func: Callable[[], Any], repeat: int) -> TimingStats:
    samples: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        _ = func()
        samples.append(time.perf_counter() - start)
    return TimingStats.from_samples(samples)


def bytes_per_second(length: int, stats: TimingStats) -> float:
    return (
        float(length) / stats.mean_seconds if stats.mean_seconds > 0 else float("inf")
    )


def values_per_second(count: int, stats: TimingStats) -> float:
    return float(count) / stats.mean_seconds if stats.mean_seconds > 0 else float("inf")


def bench_cold_start(seed: int) -> dict[str, Any]:
    methods = ["lsb", "threshold", "poincare"]
    results: dict[str, Any] = {}

    for method in methods:
        start = time.perf_counter()
        rng = ThreeBodyRNG(seed=seed, extraction_method=method)
        _ = rng.random(1)
        elapsed = time.perf_counter() - start
        results[method] = {"time_to_first_output_seconds": elapsed}

    return results


def bench_chaos_warm_random(
    seed: int, method: str, batch_sizes: list[int], repeat: int
) -> dict[str, Any]:
    rng = ThreeBodyRNG(seed=seed, extraction_method=method)
    _ = rng.random(4)  # warm-up compile

    rows: list[dict[str, Any]] = []
    for n in batch_sizes:
        stats = bench_timed(lambda n=n: rng.random(n), repeat)
        rows.append(
            {
                "batch_size": n,
                "timing": asdict(stats),
                "values_per_second": values_per_second(n, stats),
            }
        )
    return {"operation": "random", "method": method, "results": rows}


def bench_chaos_bytes(seed: int, lengths: list[int], repeat: int) -> dict[str, Any]:
    rng = ThreeBodyRNG(seed=seed)
    _ = rng.bytes(4)  # warm-up compile

    rows: list[dict[str, Any]] = []
    for n_bytes in lengths:
        stats = bench_timed(lambda n_bytes=n_bytes: rng.bytes(n_bytes), repeat)
        rows.append(
            {
                "length_bytes": n_bytes,
                "timing": asdict(stats),
                "bytes_per_second": bytes_per_second(n_bytes, stats),
            }
        )
    return {"operation": "bytes", "method": "lsb", "results": rows}


def bench_numpy_baseline(
    seed: int, batch_sizes: list[int], repeat: int
) -> dict[str, Any]:
    gen = np.random.default_rng(seed)
    _ = gen.random(4)

    rows: list[dict[str, Any]] = []
    for n in batch_sizes:
        stats = bench_timed(lambda n=n: gen.random(n), repeat)
        rows.append(
            {
                "batch_size": n,
                "timing": asdict(stats),
                "values_per_second": values_per_second(n, stats),
            }
        )
    return {
        "operation": "random",
        "baseline": "numpy.random.default_rng",
        "results": rows,
    }


def bench_python_random_baseline(
    seed: int, batch_sizes: list[int], repeat: int
) -> dict[str, Any]:
    py_rng = random.Random(seed)

    rows: list[dict[str, Any]] = []
    for n in batch_sizes:
        stats = bench_timed(lambda n=n: [py_rng.random() for _ in range(n)], repeat)
        rows.append(
            {
                "batch_size": n,
                "timing": asdict(stats),
                "values_per_second": values_per_second(n, stats),
            }
        )
    return {"operation": "random", "baseline": "random.Random", "results": rows}


def bench_byte_baselines(lengths: list[int], repeat: int) -> list[dict[str, Any]]:
    funcs: list[tuple[str, Callable[[int], bytes]]] = [
        ("os.urandom", os.urandom),
        ("secrets.token_bytes", secrets.token_bytes),
    ]

    outputs: list[dict[str, Any]] = []
    for name, func in funcs:
        rows: list[dict[str, Any]] = []
        for n_bytes in lengths:
            stats = bench_timed(lambda n=n_bytes, f=func: f(n), repeat)
            rows.append(
                {
                    "length_bytes": n_bytes,
                    "timing": asdict(stats),
                    "bytes_per_second": bytes_per_second(n_bytes, stats),
                }
            )
        outputs.append({"operation": "bytes", "baseline": name, "results": rows})

    return outputs


def bench_numpy_integration(
    seed: int, batch_sizes: list[int], repeat: int
) -> dict[str, Any]:
    gen = create_chaos_generator(seed=seed)
    _ = gen.normal(size=8)  # warm-up

    rows: list[dict[str, Any]] = []
    for n in batch_sizes:
        stats = bench_timed(lambda n=n: gen.normal(size=n), repeat)
        rows.append(
            {
                "batch_size": n,
                "timing": asdict(stats),
                "values_per_second": values_per_second(n, stats),
            }
        )
    return {
        "operation": "normal",
        "baseline": "chaos_generator_numpy_api",
        "results": rows,
    }


def get_environment() -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "numba": __import__("numba").__version__,
    }


def print_summary(data: dict[str, Any]) -> None:
    cold = data["benchmarks"]["cold_start"]
    warm_random = data["benchmarks"]["chaos_random"]["lsb"]["results"]
    numpy_random = data["benchmarks"]["baselines"]["numpy_random"]["results"]

    print("Cold start (time-to-first-output, seconds):")
    for method, row in cold.items():
        print(f"  {method:10s} {row['time_to_first_output_seconds']:.6f}")

    print()
    print("Warm throughput snapshot (values/sec):")
    print("  Scenario                Batch      Throughput")
    for row in warm_random:
        print(
            f"  chaos-rng random(lsb)   {row['batch_size']:>8}  {row['values_per_second']:>12.1f}"
        )
    for row in numpy_random:
        print(
            f"  numpy.default_rng       {row['batch_size']:>8}  {row['values_per_second']:>12.1f}"
        )

    print()
    print("Report methodology reminder:")
    print("  - Keep cold-start and warm throughput separate.")
    print("  - Include environment details when sharing numbers.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reproducible chaos-rng benchmarks"
    )
    parser.add_argument(
        "--profile",
        choices=["quick", "full"],
        default="quick",
        help="Benchmark profile. 'quick' is suitable for README and CI notes.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=None,
        help="Override repeat count for timed measurements.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for deterministic benchmark setup.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/latest.json"),
        help="Where to write JSON output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.profile == "quick":
        batch_sizes = [1_000, 100_000]
        byte_lengths = [32, 4_096]
        repeat = 3
    else:
        batch_sizes = [1_000, 10_000, 100_000, 1_000_000]
        byte_lengths = [32, 4_096, 1_048_576]
        repeat = 5

    if args.repeat is not None:
        repeat = args.repeat

    data: dict[str, Any] = {
        "schema_version": 1,
        "profile": args.profile,
        "repeat": repeat,
        "seed": args.seed,
        "environment": get_environment(),
        "benchmarks": {
            "cold_start": bench_cold_start(args.seed),
            "chaos_random": {
                method: bench_chaos_warm_random(args.seed, method, batch_sizes, repeat)
                for method in ["lsb", "threshold", "poincare"]
            },
            "chaos_bytes": bench_chaos_bytes(args.seed, byte_lengths, repeat),
            "chaos_numpy_generator": bench_numpy_integration(
                args.seed, batch_sizes, repeat
            ),
            "baselines": {
                "numpy_random": bench_numpy_baseline(args.seed, batch_sizes, repeat),
                "python_random": bench_python_random_baseline(
                    args.seed, batch_sizes, repeat
                ),
                "byte_sources": bench_byte_baselines(byte_lengths, repeat),
            },
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2), encoding="utf-8")

    print_summary(data)
    print()
    print(f"Wrote benchmark results to {args.output}")


if __name__ == "__main__":
    main()
