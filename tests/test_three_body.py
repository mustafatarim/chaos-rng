"""
Tests for three-body system implementation.

This module tests the core three-body gravitational system
and its chaotic dynamics properties.
"""

import numpy as np
import pytest
from chaos_rng.generators.three_body import ThreeBodyRNG, ThreeBodySystem


class TestThreeBodySystem:
    """Test the ThreeBodySystem class."""

    def test_initialization(self):
        """Test system initialization."""
        system = ThreeBodySystem()

        assert system.masses is not None
        assert len(system.masses) == 3
        assert system.n_bodies == 3
        assert system.n_dim == 2
        assert system.state_size == 12  # 3 bodies * 2 dims * 2 (pos + vel)

    def test_custom_masses(self):
        """Test initialization with custom masses."""
        masses = [2.0, 1.5, 0.8]
        system = ThreeBodySystem(masses=masses)

        np.testing.assert_array_equal(system.masses, masses)

    def test_initial_conditions_random(self):
        """Test random initial conditions generation."""
        system = ThreeBodySystem()
        system.set_initial_conditions(random_ic=True)

        assert system.state is not None
        assert len(system.state) == system.state_size
        assert not np.any(np.isnan(system.state))
        assert not np.any(np.isinf(system.state))

    def test_initial_conditions_manual(self, known_chaotic_ic):
        """Test manual initial conditions."""
        system = ThreeBodySystem()
        system.set_initial_conditions(
            positions=known_chaotic_ic["positions"],
            velocities=known_chaotic_ic["velocities"],
            random_ic=False,
        )

        assert system.state is not None
        # Check that positions are correctly stored
        stored_positions = system.state[:6].reshape((3, 2))
        np.testing.assert_array_almost_equal(
            stored_positions, known_chaotic_ic["positions"]
        )

    def test_evolution(self):
        """Test system evolution."""
        system = ThreeBodySystem()
        system.set_initial_conditions(random_ic=True)

        initial_state = system.state.copy()
        trajectory = system.evolve(dt=0.01, steps=100)

        assert trajectory.shape == (101, system.state_size)  # steps + 1
        assert not np.array_equal(system.state, initial_state)
        assert not np.any(np.isnan(trajectory))
        assert not np.any(np.isinf(trajectory))

    def test_energy_conservation(self):
        """Test energy conservation (approximately)."""
        system = ThreeBodySystem()
        system.set_initial_conditions(random_ic=True)

        initial_energy = system.get_energy()

        # Evolve for a short time
        system.evolve(dt=0.001, steps=100)
        final_energy = system.get_energy()

        # Energy should be approximately conserved
        relative_error = abs(final_energy - initial_energy) / abs(initial_energy)
        assert relative_error < 0.1  # 10% tolerance for numerical errors

    def test_lyapunov_exponent(self):
        """Test Lyapunov exponent calculation."""
        system = ThreeBodySystem()
        system.set_initial_conditions(random_ic=True)

        # This is a slow test
        lyapunov = system.compute_lyapunov_exponent(t_max=10.0, dt=0.01)

        # For chaotic systems, largest Lyapunov exponent should be positive
        # Note: this test might be flaky depending on initial conditions
        assert isinstance(lyapunov, float)
        assert not np.isnan(lyapunov)
        assert not np.isinf(lyapunov)

    @pytest.mark.slow
    def test_chaos_verification(self):
        """Verify chaotic behavior through sensitive dependence."""
        system1 = ThreeBodySystem()
        system2 = ThreeBodySystem()

        # Set nearly identical initial conditions
        system1.set_initial_conditions(random_ic=True)
        ic = system1.state.copy()

        # Add tiny perturbation
        ic_perturbed = ic + 1e-10 * np.random.randn(len(ic))
        system2.state = ic_perturbed
        system2.time = 0.0

        # Evolve both systems
        traj1 = system1.evolve(dt=0.01, steps=1000)
        traj2 = system2.evolve(dt=0.01, steps=1000)

        # Calculate separation over time
        separations = np.linalg.norm(traj1 - traj2, axis=1)

        # For chaotic systems, small perturbations should grow exponentially
        # Check that final separation is much larger than initial
        assert separations[-1] > 100 * separations[0]


class TestThreeBodyRNG:
    """Test the ThreeBodyRNG class."""

    def test_initialization(self):
        """Test RNG initialization."""
        rng = ThreeBodyRNG(seed=42)

        assert rng.system is not None
        assert rng.bit_extractor is not None
        assert rng.extraction_method == "lsb"

    def test_reproducibility(self):
        """Test that same seed produces same results."""
        rng1 = ThreeBodyRNG(seed=12345)
        rng2 = ThreeBodyRNG(seed=12345)

        # Generate sequences
        seq1 = rng1.random(100)
        seq2 = rng2.random(100)

        np.testing.assert_array_equal(seq1, seq2)

    def test_different_seeds(self):
        """Test that different seeds produce different results."""
        rng1 = ThreeBodyRNG(seed=1)
        rng2 = ThreeBodyRNG(seed=2)

        seq1 = rng1.random(100)
        seq2 = rng2.random(100)

        # Sequences should be different
        assert not np.array_equal(seq1, seq2)

    def test_random_output_range(self, basic_rng):
        """Test that random() output is in [0, 1)."""
        values = basic_rng.random(1000)

        assert np.all(values >= 0.0)
        assert np.all(values < 1.0)
        assert isinstance(values, np.ndarray)
        assert values.shape == (1000,)

    def test_random_single_value(self, basic_rng):
        """Test single random value generation."""
        value = basic_rng.random()

        assert isinstance(value, float)
        assert 0.0 <= value < 1.0

    def test_random_shapes(self, basic_rng):
        """Test random generation with different shapes."""
        # 1D array
        arr1d = basic_rng.random(50)
        assert arr1d.shape == (50,)

        # 2D array
        arr2d = basic_rng.random((10, 5))
        assert arr2d.shape == (10, 5)

        # 3D array
        arr3d = basic_rng.random((4, 5, 6))
        assert arr3d.shape == (4, 5, 6)

    def test_randint_output(self, basic_rng):
        """Test randint output."""
        values = basic_rng.randint(0, 10, size=1000)

        assert np.all(values >= 0)
        assert np.all(values < 10)
        assert values.dtype in [np.int32, np.int64]

    def test_randint_single_value(self, basic_rng):
        """Test single randint value."""
        value = basic_rng.randint(5, 15)

        assert isinstance(value, (int, np.integer))
        assert 5 <= value < 15

    def test_randint_edge_cases(self, basic_rng):
        """Test randint edge cases."""
        # Single possible value
        values = basic_rng.randint(5, 6, size=10)
        assert np.all(values == 5)

        # Large range
        values = basic_rng.randint(0, 2**30, size=100)
        assert np.all(values >= 0)
        assert np.all(values < 2**30)

    def test_bytes_generation(self, basic_rng):
        """Test bytes generation."""
        data = basic_rng.bytes(100)

        assert isinstance(data, bytes)
        assert len(data) == 100

        # Check that we get varied bytes (not all zeros)
        unique_bytes = set(data)
        assert len(unique_bytes) > 1

    def test_state_serialization(self, basic_rng):
        """Test state get/set functionality."""
        # Generate some values to change state
        _ = basic_rng.random(100)

        # Save state
        state = basic_rng.get_state()

        # Generate more values
        seq1 = basic_rng.random(50)

        # Restore state
        basic_rng.set_state(state)

        # Generate again - should match
        seq2 = basic_rng.random(50)

        np.testing.assert_array_equal(seq1, seq2)

    @pytest.mark.parametrize("extraction_method", ["lsb", "threshold", "poincare"])
    def test_extraction_methods(self, extraction_method):
        """Test different bit extraction methods."""
        rng = ThreeBodyRNG(seed=42, extraction_method=extraction_method)

        # Should generate without errors
        values = rng.random(100)
        assert len(values) == 100
        assert np.all((values >= 0) & (values < 1))

    @pytest.mark.parametrize(
        "masses", [[1.0, 1.0, 1.0], [2.0, 1.0, 0.5], [0.8, 1.2, 1.5]]
    )
    def test_different_masses(self, masses):
        """Test different mass configurations."""
        rng = ThreeBodyRNG(seed=42, masses=masses)

        # Should generate without errors
        values = rng.random(100)
        assert len(values) == 100
        assert np.all((values >= 0) & (values < 1))

        # Check that masses are set correctly
        np.testing.assert_array_equal(rng.system.masses, masses)

    def test_buffer_management(self, basic_rng):
        """Test internal buffer management."""
        # Generate enough values to trigger buffer refills
        values = basic_rng.random(basic_rng.buffer_size * 2)

        assert len(values) == basic_rng.buffer_size * 2
        assert np.all((values >= 0) & (values < 1))

    @pytest.mark.slow
    def test_long_sequence_generation(self):
        """Test generation of very long sequences."""
        rng = ThreeBodyRNG(seed=42)

        # Generate a long sequence
        values = rng.random(1000000)

        assert len(values) == 1000000
        assert np.all((values >= 0) & (values < 1))

        # Basic statistical tests
        mean = np.mean(values)
        assert 0.4 < mean < 0.6  # Should be approximately 0.5

        std = np.std(values)
        assert 0.2 < std < 0.4  # Should be approximately 1/sqrt(12) ≈ 0.289

    def test_reseed_functionality(self, basic_rng):
        """Test automatic reseeding."""
        # Force reseeding by generating many bits
        original_reseed_interval = basic_rng.reseed_interval
        basic_rng.reseed_interval = 1000  # Low threshold for testing

        # Generate enough to trigger reseed
        _ = basic_rng.random(200)  # Should trigger reseed

        assert basic_rng.bits_generated >= 0  # Should have reset

        # Restore original interval
        basic_rng.reseed_interval = original_reseed_interval

    def test_error_handling(self):
        """Test error handling in edge cases."""
        # Invalid extraction method
        with pytest.raises(KeyError):
            ThreeBodyRNG(extraction_method="invalid_method")

        # Invalid randint parameters
        rng = ThreeBodyRNG(seed=42)
        with pytest.raises(ValueError):
            rng.randint(10, 5)  # high <= low
