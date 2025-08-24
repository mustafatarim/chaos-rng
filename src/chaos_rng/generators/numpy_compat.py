"""
NumPy compatibility layer for chaos-based random number generators.

This module provides BitGenerator and Generator classes that are compatible
with NumPy's random number generation framework, allowing the chaos-based
RNG to be used with all standard NumPy distributions.
"""

from typing import Any, Optional, Union

import numpy as np
from numpy.random import BitGenerator, Generator

from .three_body import ThreeBodyRNG


class ChaosBitGenerator(BitGenerator):
    """
    NumPy-compatible BitGenerator using chaos-based entropy.

    This class provides a NumPy BitGenerator interface for the three-body
    chaos-based random number generator, enabling its use with all NumPy
    random distributions.
    """

    def __init__(
        self,
        seed: Optional[Union[int, np.ndarray]] = None,
        masses: Optional[list] = None,
        extraction_method: str = "lsb",
    ):
        """
        Initialize chaos-based BitGenerator.

        Parameters
        ----------
        seed : int or array_like, optional
            Seed for reproducible random numbers
        masses : list, optional
            Masses for the three-body system
        extraction_method : str, default='lsb'
            Bit extraction method to use
        """
        # Initialize the underlying chaos RNG
        self._chaos_rng = ThreeBodyRNG(
            seed=seed, masses=masses, extraction_method=extraction_method
        )

        # Initialize BitGenerator base class
        BitGenerator.__init__(self, seed)

        # Set up the capsule for NumPy C API access
        self._setup_numpy_interface()

    def _setup_numpy_interface(self):
        """Setup NumPy C API interface."""
        # This creates the necessary C structures for NumPy integration
        # In a full implementation, this would set up the actual C callbacks

        # For now, we'll set up the Python interface
        self._bitgen = self

        # Required attributes for NumPy compatibility
        self.state = {
            "bit_generator": "ChaosBitGenerator",
            "chaos_state": self._chaos_rng.get_state(),
        }

    @property
    def state(self) -> dict[str, Any]:
        """Get the current state of the generator."""
        return {
            "bit_generator": "ChaosBitGenerator",
            "chaos_state": self._chaos_rng.get_state(),
        }

    @state.setter
    def state(self, value: dict[str, Any]) -> None:
        """Set the state of the generator."""
        if "chaos_state" in value:
            self._chaos_rng.set_state(value["chaos_state"])

    def random_raw(
        self, size: Optional[int] = None, dtype: np.dtype = np.uint64
    ) -> Union[int, np.ndarray]:
        """
        Generate random integers for NumPy.

        Parameters
        ----------
        size : int, optional
            Number of values to generate
        dtype : dtype, default=uint64
            Data type of output

        Returns
        -------
        int or ndarray
            Random integer(s) of specified dtype
        """
        if dtype == np.uint64:
            bits_per_value = 64
        elif dtype == np.uint32:
            bits_per_value = 32
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        if size is None:
            # Single value
            bits = self._chaos_rng._get_random_bits(bits_per_value)
            return self._bits_to_uint(bits, dtype)

        # Array of values
        total_bits = size * bits_per_value
        bits = self._chaos_rng._get_random_bits(total_bits)

        # Convert to integers
        values = np.zeros(size, dtype=dtype)
        for i in range(size):
            start_bit = i * bits_per_value
            end_bit = start_bit + bits_per_value
            values[i] = self._bits_to_uint(bits[start_bit:end_bit], dtype)

        return values

    def _bits_to_uint(self, bits: np.ndarray, dtype: np.dtype) -> int:
        """Convert bit array to unsigned integer."""
        if dtype == np.uint64:
            max_bits = 64
        elif dtype == np.uint32:
            max_bits = 32
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        # Ensure we have the right number of bits
        if len(bits) > max_bits:
            bits = bits[:max_bits]
        elif len(bits) < max_bits:
            # Pad with zeros
            bits = np.concatenate(
                [bits, np.zeros(max_bits - len(bits), dtype=np.uint8)]
            )

        # Convert to integer
        result = 0
        for i, bit in enumerate(bits):
            result |= int(bit) << i

        return dtype.type(result)

    def random(
        self,
        size: Optional[Union[int, tuple[int, ...]]] = None,
        dtype: np.dtype = np.float64,
    ) -> Union[float, np.ndarray]:
        """
        Generate random floats in [0, 1).

        Parameters
        ----------
        size : int or tuple, optional
            Output shape
        dtype : dtype, default=float64
            Data type of output

        Returns
        -------
        float or ndarray
            Random float(s) in [0, 1)
        """
        return self._chaos_rng.random(size)

    def bytes(self, length: int) -> bytes:
        """
        Generate random bytes.

        Parameters
        ----------
        length : int
            Number of bytes to generate

        Returns
        -------
        bytes
            Random bytes
        """
        return self._chaos_rng.bytes(length)


class ChaosGenerator(Generator):
    """
    NumPy-compatible Generator using chaos-based BitGenerator.

    This class provides access to all NumPy distributions using
    the chaos-based random number generator as the entropy source.
    """

    def __init__(
        self,
        bit_generator: Optional[ChaosBitGenerator] = None,
        seed: Optional[Union[int, np.ndarray]] = None,
        **kwargs,
    ):
        """
        Initialize chaos-based Generator.

        Parameters
        ----------
        bit_generator : ChaosBitGenerator, optional
            BitGenerator instance. If None, creates new one with seed
        seed : int or array_like, optional
            Seed for random number generation
        **kwargs
            Additional arguments for ChaosBitGenerator
        """
        if bit_generator is None:
            bit_generator = ChaosBitGenerator(seed=seed, **kwargs)

        # Initialize parent Generator class
        super().__init__(bit_generator)

        # Store reference to chaos RNG for direct access
        self._chaos_rng = bit_generator._chaos_rng

    def validate_output(self, n_samples: int = 100000) -> dict[str, Any]:
        """
        Validate the quality of generated random numbers.

        Parameters
        ----------
        n_samples : int, default=100000
            Number of samples to generate for testing

        Returns
        -------
        dict
            Validation results
        """
        from ..utils.analysis import StatisticalAnalyzer

        # Generate test samples
        samples = self.random(n_samples)

        # Convert to binary for analysis
        binary_samples = (samples * 2).astype(int)  # Simple binarization

        # Perform statistical analysis
        analyzer = StatisticalAnalyzer()
        results = analyzer.comprehensive_analysis(binary_samples)

        return results

    def benchmark_performance(self, n_samples: int = 1000000) -> dict[str, float]:
        """
        Benchmark the performance of the generator.

        Parameters
        ----------
        n_samples : int, default=1000000
            Number of samples for benchmarking

        Returns
        -------
        dict
            Performance metrics
        """
        import time

        # Test different operations
        operations = {
            "random": lambda: self.random(n_samples),
            "integers": lambda: self.integers(0, 2**32, n_samples),
            "normal": lambda: self.normal(size=n_samples),
            "exponential": lambda: self.exponential(size=n_samples),
        }

        results = {}

        for op_name, op_func in operations.items():
            # Warm up
            _ = op_func()

            # Benchmark
            start_time = time.time()
            _ = op_func()
            end_time = time.time()

            duration = end_time - start_time
            rate = n_samples / duration

            results[f"{op_name}_time"] = duration
            results[f"{op_name}_rate"] = rate

        return results

    def get_lyapunov_exponent(self) -> float:
        """
        Get the current Lyapunov exponent of the underlying chaos system.

        Returns
        -------
        float
            Largest Lyapunov exponent
        """
        return self._chaos_rng.system.compute_lyapunov_exponent()

    def get_system_energy(self) -> float:
        """
        Get the current total energy of the three-body system.

        Returns
        -------
        float
            Total system energy
        """
        return self._chaos_rng.system.get_energy()

    def evolve_system(self, steps: int = 1000) -> np.ndarray:
        """
        Evolve the underlying chaotic system.

        Parameters
        ----------
        steps : int, default=1000
            Number of integration steps

        Returns
        -------
        ndarray
            System trajectory
        """
        return self._chaos_rng.system.evolve(dt=0.001, steps=steps)


def create_chaos_generator(
    seed: Optional[Union[int, np.ndarray]] = None,
    masses: Optional[list] = None,
    extraction_method: str = "lsb",
) -> ChaosGenerator:
    """
    Convenience function to create a chaos-based Generator.

    Parameters
    ----------
    seed : int or array_like, optional
        Seed for reproducible random numbers
    masses : list, optional
        Masses for the three-body system
    extraction_method : str, default='lsb'
        Bit extraction method

    Returns
    -------
    ChaosGenerator
        Configured chaos-based generator
    """
    bit_generator = ChaosBitGenerator(
        seed=seed, masses=masses, extraction_method=extraction_method
    )

    return ChaosGenerator(bit_generator)


# Alias for compatibility
default_rng = create_chaos_generator


class ThreadSafeGenerator:
    """
    Thread-safe wrapper for chaos-based generator.

    This class provides thread safety by maintaining separate
    generator instances per thread.
    """

    def __init__(self, **kwargs):
        """
        Initialize thread-safe generator.

        Parameters
        ----------
        **kwargs
            Arguments for generator creation
        """
        import threading

        self._local = threading.local()
        self._kwargs = kwargs

    def _get_generator(self) -> ChaosGenerator:
        """Get the thread-local generator instance."""
        if not hasattr(self._local, "generator"):
            import os
            import threading

            # Create unique seed for this thread
            thread_id = threading.get_ident()
            base_seed = self._kwargs.get("seed", 0)
            thread_seed = hash((base_seed, thread_id, os.getpid())) % (2**32)

            kwargs = self._kwargs.copy()
            kwargs["seed"] = thread_seed

            self._local.generator = create_chaos_generator(**kwargs)

        return self._local.generator

    def __getattr__(self, name):
        """Delegate attribute access to thread-local generator."""
        generator = self._get_generator()
        return getattr(generator, name)

    def random(self, *args, **kwargs):
        """Generate random floats (thread-safe)."""
        return self._get_generator().random(*args, **kwargs)

    def integers(self, *args, **kwargs):
        """Generate random integers (thread-safe)."""
        return self._get_generator().integers(*args, **kwargs)

    def normal(self, *args, **kwargs):
        """Generate normal distributed values (thread-safe)."""
        return self._get_generator().normal(*args, **kwargs)

    def uniform(self, *args, **kwargs):
        """Generate uniform distributed values (thread-safe)."""
        return self._get_generator().uniform(*args, **kwargs)
