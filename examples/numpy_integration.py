#!/usr/bin/env python3
"""Demonstrate NumPy-compatible generator usage."""

import numpy as np

from chaos_rng.generators import create_chaos_generator


def main() -> None:
    gen = create_chaos_generator(seed=42)

    normal = gen.normal(size=1_000)
    uniform = gen.uniform(0, 10, size=1_000)
    integers = gen.integers(0, 100, size=20)

    print("normal mean/std:", float(np.mean(normal)), float(np.std(normal)))
    print("uniform min/max:", float(np.min(uniform)), float(np.max(uniform)))
    print("integers sample:", integers[:10])


if __name__ == "__main__":
    main()
