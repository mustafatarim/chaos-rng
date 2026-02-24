"""
Chaos-based Random Number Generator Library

A Python library for generating random numbers using three-body problem
dynamics and other chaotic systems.

The library also includes optional NumPy integration and helper utilities
for validation and cryptographic post-processing experiments.
"""

__version__ = "0.1.4"
__author__ = "Mustafa Tarim"
__email__ = "mail@mustafatarim.com"

from .generators.bit_extractors import BitExtractor
from .generators.numpy_compat import (
    ChaosBitGenerator,
    ChaosGenerator,
    create_chaos_generator,
)
from .generators.three_body import ThreeBodyRNG, ThreeBodySystem

__all__ = [
    "ThreeBodyRNG",
    "ThreeBodySystem",
    "BitExtractor",
    "ChaosBitGenerator",
    "ChaosGenerator",
    "create_chaos_generator",
    "__version__",
]
