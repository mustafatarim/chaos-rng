"""
Chaos-based Random Number Generator Library

A professional Python library for generating high-quality random numbers
using three-body problem dynamics and other chaotic systems.

This library provides cryptographically secure random number generation
using the inherent unpredictability of chaotic dynamical systems.
"""

__version__ = "0.1.0"
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
