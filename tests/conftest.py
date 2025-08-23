"""
Pytest configuration and fixtures for chaos-rng tests.

This module provides common fixtures and configuration for the test suite.
"""

import pytest
import numpy as np
from chaos_rng import ThreeBodyRNG
from chaos_rng.generators.numpy_compat import ChaosBitGenerator, ChaosGenerator


@pytest.fixture
def basic_rng():
    """Create a basic ThreeBodyRNG instance for testing."""
    return ThreeBodyRNG(seed=42)


@pytest.fixture
def reproducible_rng():
    """Create a reproducible RNG with fixed seed."""
    return ThreeBodyRNG(seed=12345, extraction_method='lsb')


@pytest.fixture
def chaos_bitgen():
    """Create a ChaosBitGenerator for NumPy compatibility tests."""
    return ChaosBitGenerator(seed=42)


@pytest.fixture
def chaos_generator():
    """Create a ChaosGenerator for distribution tests."""
    return ChaosGenerator(seed=42)


@pytest.fixture
def test_sequence_small():
    """Generate a small test sequence for quick tests."""
    rng = ThreeBodyRNG(seed=999)
    return (rng.random(1000) * 2).astype(int)  # Binary sequence


@pytest.fixture
def test_sequence_large():
    """Generate a large test sequence for statistical tests."""
    rng = ThreeBodyRNG(seed=999)
    return (rng.random(100000) * 2).astype(int)  # Binary sequence


@pytest.fixture(params=[
    {'masses': [1.0, 1.0, 1.0], 'extraction_method': 'lsb'},
    {'masses': [1.0, 2.0, 0.5], 'extraction_method': 'threshold'},
    {'masses': [2.0, 1.5, 1.2], 'extraction_method': 'poincare'},
])
def parametrized_rng(request):
    """Parametrized RNG fixture for testing different configurations."""
    return ThreeBodyRNG(seed=42, **request.param)


@pytest.fixture
def temp_file(tmp_path):
    """Create a temporary file for testing file operations."""
    return tmp_path / "test_output.bin"


# Test data for specific scenarios
@pytest.fixture
def known_chaotic_ic():
    """Initial conditions known to produce chaotic behavior."""
    return {
        'positions': np.array([
            [-1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0]
        ]),
        'velocities': np.array([
            [0.0, 0.5],
            [0.0, -0.5],
            [-0.5, 0.0]
        ])
    }


@pytest.fixture
def stable_ic():
    """Initial conditions for relatively stable motion."""
    return {
        'positions': np.array([
            [-2.0, 0.0],
            [2.0, 0.0],
            [0.0, 2.0]
        ]),
        'velocities': np.array([
            [0.0, 0.1],
            [0.0, -0.1],
            [-0.1, 0.0]
        ])
    }


# Markers for different test categories
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "statistical: marks tests that perform statistical validation"
    )
    config.addinivalue_line(
        "markers", "nist: marks tests that require NIST test suite"
    )


# Test configuration
TEST_CONFIG = {
    'small_sample_size': 1000,
    'medium_sample_size': 10000,
    'large_sample_size': 100000,
    'statistical_threshold': 0.01,
    'performance_iterations': 100,
    'lyapunov_tolerance': 0.1
}


@pytest.fixture
def test_config():
    """Provide test configuration parameters."""
    return TEST_CONFIG