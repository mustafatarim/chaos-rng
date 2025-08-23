"""
Tests for NumPy compatibility layer.

This module tests the NumPy BitGenerator and Generator compatibility
for the chaos-based random number generator.
"""

import pytest
import numpy as np
from chaos_rng.generators.numpy_compat import (
    ChaosBitGenerator, ChaosGenerator, create_chaos_generator,
    ThreadSafeGenerator
)


class TestChaosBitGenerator:
    """Test the ChaosBitGenerator class."""
    
    def test_initialization(self):
        """Test BitGenerator initialization."""
        bitgen = ChaosBitGenerator(seed=42)
        
        assert bitgen is not None
        assert hasattr(bitgen, '_chaos_rng')
        assert bitgen.state is not None
    
    def test_reproducibility(self):
        """Test reproducible output with same seed."""
        bitgen1 = ChaosBitGenerator(seed=12345)
        bitgen2 = ChaosBitGenerator(seed=12345)
        
        # Generate raw random integers
        values1 = bitgen1.random_raw(10)
        values2 = bitgen2.random_raw(10)
        
        np.testing.assert_array_equal(values1, values2)
    
    def test_different_seeds(self):
        """Test different output with different seeds."""
        bitgen1 = ChaosBitGenerator(seed=1)
        bitgen2 = ChaosBitGenerator(seed=2)
        
        values1 = bitgen1.random_raw(10)
        values2 = bitgen2.random_raw(10)
        
        # Should be different
        assert not np.array_equal(values1, values2)
    
    def test_random_raw_uint64(self):
        """Test raw random generation with uint64."""
        bitgen = ChaosBitGenerator(seed=42)
        
        # Single value
        value = bitgen.random_raw()
        assert isinstance(value, (int, np.uint64))
        
        # Array of values
        values = bitgen.random_raw(100)
        assert values.shape == (100,)
        assert values.dtype == np.uint64
    
    def test_random_raw_uint32(self):
        """Test raw random generation with uint32."""
        bitgen = ChaosBitGenerator(seed=42)
        
        values = bitgen.random_raw(50, dtype=np.uint32)
        assert values.shape == (50,)
        assert values.dtype == np.uint32
    
    def test_random_floats(self):
        """Test random float generation."""
        bitgen = ChaosBitGenerator(seed=42)
        
        # Single value
        value = bitgen.random()
        assert isinstance(value, float)
        assert 0.0 <= value < 1.0
        
        # Array of values
        values = bitgen.random(100)
        assert values.shape == (100,)
        assert np.all((values >= 0.0) & (values < 1.0))
    
    def test_bytes_generation(self):
        """Test random bytes generation."""
        bitgen = ChaosBitGenerator(seed=42)
        
        data = bitgen.bytes(100)
        assert isinstance(data, bytes)
        assert len(data) == 100
    
    def test_state_management(self):
        """Test state get/set functionality."""
        bitgen = ChaosBitGenerator(seed=42)
        
        # Generate some values to change state
        _ = bitgen.random_raw(10)
        
        # Save state
        state = bitgen.state
        
        # Generate more values
        values1 = bitgen.random_raw(10)
        
        # Restore state
        bitgen.state = state
        
        # Generate again - should match
        values2 = bitgen.random_raw(10)
        
        np.testing.assert_array_equal(values1, values2)
    
    @pytest.mark.parametrize("extraction_method", ['lsb', 'threshold', 'poincare'])
    def test_extraction_methods(self, extraction_method):
        """Test different extraction methods."""
        bitgen = ChaosBitGenerator(seed=42, extraction_method=extraction_method)
        
        values = bitgen.random_raw(100)
        assert len(values) == 100
    
    def test_custom_masses(self):
        """Test custom masses for three-body system."""
        masses = [2.0, 1.5, 0.8]
        bitgen = ChaosBitGenerator(seed=42, masses=masses)
        
        values = bitgen.random_raw(100)
        assert len(values) == 100
        
        # Check that masses are set correctly
        np.testing.assert_array_equal(bitgen._chaos_rng.system.masses, masses)


class TestChaosGenerator:
    """Test the ChaosGenerator class."""
    
    def test_initialization(self):
        """Test Generator initialization."""
        gen = ChaosGenerator(seed=42)
        
        assert gen is not None
        assert hasattr(gen, 'bit_generator')
        assert isinstance(gen.bit_generator, ChaosBitGenerator)
    
    def test_numpy_distributions(self):
        """Test that NumPy distributions work."""
        gen = ChaosGenerator(seed=42)
        
        # Normal distribution
        normal_samples = gen.normal(size=100)
        assert normal_samples.shape == (100,)
        assert not np.all(normal_samples == normal_samples[0])  # Should vary
        
        # Uniform distribution
        uniform_samples = gen.uniform(0, 10, size=50)
        assert uniform_samples.shape == (50,)
        assert np.all((uniform_samples >= 0) & (uniform_samples < 10))
        
        # Integer distribution
        int_samples = gen.integers(0, 100, size=75)
        assert int_samples.shape == (75,)
        assert np.all((int_samples >= 0) & (int_samples < 100))
        
        # Exponential distribution
        exp_samples = gen.exponential(scale=2.0, size=80)
        assert exp_samples.shape == (80,)
        assert np.all(exp_samples >= 0)
    
    def test_reproducibility_with_distributions(self):
        """Test reproducibility with NumPy distributions."""
        gen1 = ChaosGenerator(seed=12345)
        gen2 = ChaosGenerator(seed=12345)
        
        # Normal distribution
        normal1 = gen1.normal(size=100)
        normal2 = gen2.normal(size=100)
        np.testing.assert_array_equal(normal1, normal2)
        
        # Uniform distribution
        uniform1 = gen1.uniform(size=50)
        uniform2 = gen2.uniform(size=50)
        np.testing.assert_array_equal(uniform1, uniform2)
    
    def test_validate_output(self):
        """Test output validation functionality."""
        gen = ChaosGenerator(seed=42)
        
        # This might take a while, so use smaller sample
        results = gen.validate_output(n_samples=1000)
        
        assert isinstance(results, dict)
        assert 'sequence_length' in results
        assert 'overall_quality' in results
    
    @pytest.mark.slow
    def test_benchmark_performance(self):
        """Test performance benchmarking."""
        gen = ChaosGenerator(seed=42)
        
        # Use smaller sample size for testing
        results = gen.benchmark_performance(n_samples=10000)
        
        assert isinstance(results, dict)
        assert 'random_time' in results
        assert 'random_rate' in results
        assert 'integers_time' in results
        assert 'normal_time' in results
        
        # All rates should be positive
        for key in results:
            if key.endswith('_rate'):
                assert results[key] > 0
    
    def test_chaos_specific_methods(self):
        """Test chaos-specific methods."""
        gen = ChaosGenerator(seed=42)
        
        # Lyapunov exponent
        lyapunov = gen.get_lyapunov_exponent()
        assert isinstance(lyapunov, float)
        
        # System energy
        energy = gen.get_system_energy()
        assert isinstance(energy, float)
        
        # System evolution
        trajectory = gen.evolve_system(steps=100)
        assert trajectory.shape[0] == 101  # steps + 1
        assert trajectory.shape[1] == 12   # 3 bodies * 2 dims * 2 (pos + vel)


class TestConvenienceFunctions:
    """Test convenience functions for creating generators."""
    
    def test_create_chaos_generator(self):
        """Test create_chaos_generator function."""
        gen = create_chaos_generator(seed=42)
        
        assert isinstance(gen, ChaosGenerator)
        assert gen.bit_generator.state is not None
    
    def test_create_with_parameters(self):
        """Test creation with custom parameters."""
        masses = [1.5, 1.0, 2.0]
        gen = create_chaos_generator(
            seed=42, 
            masses=masses, 
            extraction_method='threshold'
        )
        
        assert isinstance(gen, ChaosGenerator)
        np.testing.assert_array_equal(gen._chaos_rng.system.masses, masses)
        assert gen._chaos_rng.extraction_method == 'threshold'


class TestThreadSafeGenerator:
    """Test thread-safe generator functionality."""
    
    def test_initialization(self):
        """Test thread-safe generator initialization."""
        gen = ThreadSafeGenerator(seed=42)
        
        assert gen is not None
        assert hasattr(gen, '_local')
        assert hasattr(gen, '_kwargs')
    
    def test_thread_local_creation(self):
        """Test that generators are created per thread."""
        gen = ThreadSafeGenerator(seed=42)
        
        # Get generator for current thread
        thread_gen = gen._get_generator()
        
        assert isinstance(thread_gen, ChaosGenerator)
    
    def test_basic_operations(self):
        """Test basic operations work through delegation."""
        gen = ThreadSafeGenerator(seed=42)
        
        # Random floats
        values = gen.random(100)
        assert values.shape == (100,)
        assert np.all((values >= 0) & (values < 1))
        
        # Random integers
        ints = gen.integers(0, 10, size=50)
        assert ints.shape == (50,)
        assert np.all((ints >= 0) & (ints < 10))
        
        # Normal distribution
        normal = gen.normal(size=75)
        assert normal.shape == (75,)
    
    @pytest.mark.slow
    def test_thread_independence(self):
        """Test that different threads get different sequences."""
        import threading
        import queue
        
        results = queue.Queue()
        
        def worker(thread_id):
            gen = ThreadSafeGenerator(seed=42)  # Same seed
            values = gen.random(100)
            results.put((thread_id, values))
        
        # Start multiple threads
        threads = []
        for i in range(3):
            t = threading.Thread(target=worker, args=(i,))
            t.start()
            threads.append(t)
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Collect results
        thread_results = []
        while not results.empty():
            thread_results.append(results.get())
        
        assert len(thread_results) == 3
        
        # Each thread should get different sequences
        # (because thread IDs are different, seeds will be different)
        values_list = [values for _, values in thread_results]
        
        # Check that at least some pairs are different
        different_pairs = 0
        for i in range(len(values_list)):
            for j in range(i + 1, len(values_list)):
                if not np.array_equal(values_list[i], values_list[j]):
                    different_pairs += 1
        
        assert different_pairs > 0  # At least some should be different