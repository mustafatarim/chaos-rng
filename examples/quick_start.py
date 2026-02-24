#!/usr/bin/env python3
"""Minimal core API example for chaos-rng."""

from chaos_rng import ThreeBodyRNG


def main() -> None:
    rng = ThreeBodyRNG(seed=42)

    print("random(5):", rng.random(5))
    print("randint(0, 10, 5):", rng.randint(0, 10, 5))
    print("bytes(16):", rng.bytes(16).hex())

    print()
    print("Note: first call can be slower because Numba compiles kernels.")


if __name__ == "__main__":
    main()
