#!/usr/bin/env python3
"""Offline validation example (computationally heavier than quick-start)."""

from chaos_rng import ThreeBodyRNG
from chaos_rng.utils.validation import EntropyValidator


def main() -> None:
    rng = ThreeBodyRNG(seed=42)
    bits = (rng.random(10_000) * 2).astype(int)

    validator = EntropyValidator()
    results = validator.validate_entropy_rate(bits)

    print("entropy validation result keys:", sorted(results.keys()))
    print("overall valid:", results.get("overall_valid"))

    print()
    print("This example is intended for offline analysis, not a default CI check.")


if __name__ == "__main__":
    main()
