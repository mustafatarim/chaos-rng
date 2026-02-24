#!/usr/bin/env python3
"""Demonstrate deterministic output when using the same seed."""

import numpy as np

from chaos_rng import ThreeBodyRNG


def main() -> None:
    seed = 12345
    n = 10

    rng_a = ThreeBodyRNG(seed=seed)
    rng_b = ThreeBodyRNG(seed=seed)
    rng_c = ThreeBodyRNG(seed=seed + 1)

    a = rng_a.random(n)
    b = rng_b.random(n)
    c = rng_c.random(n)

    print("same-seed equal:", np.allclose(a, b))
    print("different-seed equal:", np.allclose(a, c))
    print("sample:", a[:5])

    print()
    print("Use fixed seeds for tests and experiments. Avoid fixed seeds in production.")


if __name__ == "__main__":
    main()
