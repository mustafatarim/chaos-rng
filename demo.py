#!/usr/bin/env python3
"""
Demonstration script for chaos-rng library.

This script shows the basic functionality of the chaos-based
random number generator library.
"""

import matplotlib.pyplot as plt
import numpy as np

from chaos_rng import ThreeBodyRNG, create_chaos_generator


def main():
    """Run demonstration of chaos-rng functionality."""
    print("🌌 Chaos-Based Random Number Generator Demo")
    print("=" * 50)

    # 1. Basic RNG functionality
    print("\n1. Basic Random Number Generation")
    print("-" * 35)

    rng = ThreeBodyRNG(seed=42)

    # Generate random floats
    random_floats = rng.random(10)
    print(f"Random floats: {random_floats}")

    # Generate random integers
    random_ints = rng.randint(0, 100, size=5)
    print(f"Random integers: {random_ints}")

    # Generate random bytes
    random_bytes = rng.bytes(16)
    print(f"Random bytes (hex): {random_bytes.hex()}")

    # 2. NumPy integration
    print("\n2. NumPy Integration")
    print("-" * 25)

    np_rng = create_chaos_generator(seed=42)

    # Normal distribution
    normal_samples = np_rng.normal(loc=0, scale=1, size=100)
    print(
        "Normal samples: "
        f"mean={np.mean(normal_samples):.3f}, std={np.std(normal_samples):.3f}"
    )

    # Uniform distribution
    uniform_samples = np_rng.uniform(0, 10, size=100)
    print(
        "Uniform samples: "
        f"min={np.min(uniform_samples):.3f}, max={np.max(uniform_samples):.3f}"
    )

    # 3. System properties
    print("\n3. Chaotic System Properties")
    print("-" * 32)

    system_energy = rng.system.get_energy()
    print(f"System energy: {system_energy:.6f}")

    print("Computing Lyapunov exponent (this may take a moment)...")
    lyapunov = rng.system.compute_lyapunov_exponent(t_max=10.0)
    print(f"Lyapunov exponent: {lyapunov:.6f}")

    if lyapunov > 0:
        print("✓ System exhibits chaotic behavior")
    else:
        print("⚠ System may not be chaotic with current parameters")

    # 4. Different extraction methods
    print("\n4. Bit Extraction Methods")
    print("-" * 30)

    methods = ["lsb", "threshold", "poincare"]
    for method in methods:
        method_rng = ThreeBodyRNG(seed=42, extraction_method=method)
        samples = method_rng.random(1000)
        mean = np.mean(samples)
        std = np.std(samples)
        print(f"{method:10s}: mean={mean:.4f}, std={std:.4f}")

    # 5. Statistical quality check
    print("\n5. Basic Statistical Quality")
    print("-" * 32)

    # Generate larger sample for analysis
    large_sample = rng.random(10000)

    # Basic statistics
    sample_mean = np.mean(large_sample)
    sample_std = np.std(large_sample)

    print(f"Sample size: {len(large_sample)}")
    print(f"Mean: {sample_mean:.6f} (expected ~0.5)")
    print(f"Std:  {sample_std:.6f} (expected ~0.289)")

    # Frequency test for binary version
    binary_sample = (large_sample * 2).astype(int)
    ones_freq = np.mean(binary_sample)
    print(f"Ones frequency: {ones_freq:.6f} (expected ~0.5)")

    # 6. Visualization (if matplotlib is available)
    print("\n6. Visualization")
    print("-" * 18)

    try:
        # Plot trajectory
        trajectory = rng.system.evolve(dt=0.01, steps=1000)
        positions = trajectory[:, :6].reshape(-1, 3, 2)  # 3 bodies, 2D positions

        plt.figure(figsize=(12, 4))

        # Plot phase space trajectory
        plt.subplot(131)
        for i in range(3):
            plt.plot(
                positions[:, i, 0], positions[:, i, 1], label=f"Body {i+1}", alpha=0.7
            )
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.title("Three-Body Trajectories")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot histogram of random values
        plt.subplot(132)
        plt.hist(large_sample, bins=50, alpha=0.7, density=True)
        plt.axhline(y=1.0, color="r", linestyle="--", label="Expected (uniform)")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.title("Random Values Distribution")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot autocorrelation
        plt.subplot(133)
        binary_centered = binary_sample.astype(float) - 0.5
        autocorr = np.correlate(binary_centered, binary_centered, mode="full")
        autocorr = autocorr[autocorr.size // 2 :]
        autocorr = autocorr / autocorr[0]  # Normalize

        lags = np.arange(min(100, len(autocorr)))
        plt.plot(lags, autocorr[: len(lags)])
        plt.axhline(y=0, color="r", linestyle="--", alpha=0.5)
        plt.xlabel("Lag")
        plt.ylabel("Autocorrelation")
        plt.title("Autocorrelation Function")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("chaos_rng_demo.png", dpi=150, bbox_inches="tight")
        plt.show()

        print("✓ Plots saved as 'chaos_rng_demo.png'")

    except ImportError:
        print("⚠ Matplotlib not available, skipping visualization")
    except Exception as e:
        print(f"⚠ Visualization failed: {e}")

    print("\n🎉 Demo completed successfully!")
    print("\nNext steps:")
    print("- Run the test suite: pytest tests/")
    print("- Check the documentation in README.md")
    print("- Explore advanced features in the examples/")


if __name__ == "__main__":
    main()
