"""
Bit extraction methods for converting chaotic trajectories to random bits.

This module implements various methods to extract random bits from chaotic
system trajectories, each with different statistical properties and performance
characteristics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numba
import numpy as np

from chaos_rng.utils.analysis import StatisticalAnalyzer


class BitExtractionMethod(ABC):
    """Abstract base class for bit extraction methods."""

    @abstractmethod
    def extract(self, trajectory: np.ndarray) -> np.ndarray:
        """Extract bits from trajectory data."""
        pass


class LSBExtractor(BitExtractionMethod):
    """
    Least Significant Bit (LSB) extraction method.

    This method extracts the least significant bits from floating-point
    representations of trajectory coordinates after scaling and quantization.

    Formula: bit = int(abs(coord) * 2**32) % 2
    """

    def __init__(self, scale_factor: float = 2**32):
        """
        Initialize LSB extractor.

        Parameters
        ----------
        scale_factor : float, default=2**32
            Scaling factor for quantization
        """
        self.scale_factor = scale_factor

    def extract(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Extract LSBs from trajectory.

        Parameters
        ----------
        trajectory : ndarray
            System trajectory [n_steps, state_dim]

        Returns
        -------
        ndarray
            Array of random bits
        """
        return _lsb_extract_jit(trajectory.flatten(), self.scale_factor)


class ThresholdExtractor(BitExtractionMethod):
    """
    Threshold-based bit extraction method.

    This method compares consecutive values in the trajectory to extract bits.

    Formula: bit = 1 if x(n+1) > x(n) else 0
    """

    def __init__(self, coordinate_indices: np.ndarray | None = None):
        """
        Initialize threshold extractor.

        Parameters
        ----------
        coordinate_indices : ndarray, optional
            Indices of coordinates to use. If None, use all coordinates
        """
        self.coordinate_indices = coordinate_indices

    def extract(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Extract bits using threshold method.

        Parameters
        ----------
        trajectory : ndarray
            System trajectory [n_steps, state_dim]

        Returns
        -------
        ndarray
            Array of random bits
        """
        if self.coordinate_indices is not None:
            data = trajectory[:, self.coordinate_indices]
        else:
            data = trajectory

        return _threshold_extract_jit(data)


class PoincareExtractor(BitExtractionMethod):
    """
    Poincaré section bit extraction method.

    This method samples bits when the trajectory crosses specific hyperplanes,
    providing excellent decorrelation properties.
    """

    def __init__(
        self,
        section_normal: np.ndarray | None = None,
        section_offset: float = 0.0,
        coordinate_idx: int = 0,
    ):
        """
        Initialize Poincaré extractor.

        Parameters
        ----------
        section_normal : ndarray, optional
            Normal vector to Poincaré section hyperplane
        section_offset : float, default=0.0
            Offset of the hyperplane
        coordinate_idx : int, default=0
            Coordinate index to use for simple sections
        """
        self.section_normal = section_normal
        self.section_offset = section_offset
        self.coordinate_idx = coordinate_idx

    def extract(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Extract bits using Poincaré sections.

        Parameters
        ----------
        trajectory : ndarray
            System trajectory [n_steps, state_dim]

        Returns
        -------
        ndarray
            Array of random bits
        """
        if self.section_normal is not None:
            return _poincare_extract_hyperplane_jit(
                trajectory, self.section_normal, self.section_offset
            )
        else:
            return _poincare_extract_coordinate_jit(
                trajectory, self.coordinate_idx, self.section_offset
            )


class BitExtractor:
    """
    Unified bit extractor with multiple methods and post-processing.

    This class combines different extraction methods and applies
    statistical post-processing to improve randomness quality.
    """

    def __init__(
        self, method: str = "lsb", post_process: bool = True, debias: bool = True
    ):
        """
        Initialize bit extractor.

        Parameters
        ----------
        method : str, default='lsb'
            Extraction method: 'lsb', 'threshold', 'poincare', or 'combined'
        post_process : bool, default=True
            Whether to apply statistical post-processing
        debias : bool, default=True
            Whether to apply von Neumann debiasing
        """
        self.method = method
        self.post_process = post_process
        self.debias = debias

        # Initialize extraction methods
        self._extractors = {
            "lsb": LSBExtractor(),
            "threshold": ThresholdExtractor(),
            "poincare": PoincareExtractor(),
        }

        valid_methods = set(self._extractors.keys()) | {"combined"}
        if self.method not in valid_methods:
            raise KeyError(f"Unknown extraction method: {self.method}")

    def extract_bits(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Extract random bits from trajectory using specified method.

        Parameters
        ----------
        trajectory : ndarray
            System trajectory [n_steps, state_dim]

        Returns
        -------
        ndarray
            Array of random bits
        """
        if self.method == "combined":
            # Combine multiple extraction methods
            bits = self._extract_combined(trajectory)
        else:
            # Single extraction method
            extractor = self._extractors[self.method]
            bits = extractor.extract(trajectory)

        # Apply post-processing
        if self.post_process:
            bits = self._post_process(bits)

        if self.debias:
            bits = self._von_neumann_debias(bits)

        return bits

    def _extract_combined(self, trajectory: np.ndarray) -> np.ndarray:
        """Combine multiple extraction methods for enhanced entropy."""
        bits_list = []

        # Extract with each method
        for extractor in self._extractors.values():
            try:
                method_bits = extractor.extract(trajectory)
                bits_list.append(method_bits)
            except (
                Exception
            ):  # nosec B112 - best-effort aggregation, failures are acceptable
                # Skip methods that fail
                continue

        if not bits_list:
            raise RuntimeError("All extraction methods failed")

        # Combine using XOR
        combined_bits = bits_list[0]
        for bits in bits_list[1:]:
            min_len = min(len(combined_bits), len(bits))
            combined_bits = combined_bits[:min_len] ^ bits[:min_len]

        return combined_bits

    def _post_process(self, bits: np.ndarray) -> np.ndarray:
        """Apply statistical post-processing to improve randomness."""
        return _xor_shift_postprocess_jit(bits)

    def _von_neumann_debias(self, bits: np.ndarray) -> np.ndarray:
        """Apply von Neumann debiasing to remove statistical bias."""
        return _von_neumann_debias_jit(bits)


# JIT-compiled extraction functions for maximum performance


@numba.njit(cache=True)
def _lsb_extract_jit(data: np.ndarray, scale_factor: float) -> np.ndarray:
    """
    JIT-compiled LSB extraction.

    This function extracts the least significant bits from scaled
    floating-point values with maximum performance.
    """
    n_points = len(data)
    bits = np.zeros(n_points, dtype=np.uint8)

    for i in range(n_points):
        # Scale and take absolute value
        scaled = abs(data[i]) * scale_factor

        # Extract LSB after conversion to integer
        int_val = int(scaled)
        bits[i] = int_val & 1

    return bits


@numba.njit(cache=True)
def _threshold_extract_jit(trajectory: np.ndarray) -> np.ndarray:
    """
    JIT-compiled threshold extraction.

    Compares consecutive trajectory points to generate bits.
    """
    n_steps, n_coords = trajectory.shape
    n_bits = (n_steps - 1) * n_coords
    bits = np.zeros(n_bits, dtype=np.uint8)

    bit_idx = 0
    for i in range(n_steps - 1):
        for j in range(n_coords):
            # Compare consecutive values
            if trajectory[i + 1, j] > trajectory[i, j]:
                bits[bit_idx] = 1
            else:
                bits[bit_idx] = 0
            bit_idx += 1

    return bits


@numba.njit(cache=True)
def _poincare_extract_coordinate_jit(
    trajectory: np.ndarray, coord_idx: int, threshold: float
) -> np.ndarray:
    """
    JIT-compiled Poincaré extraction using coordinate crossings.
    """
    n_steps = trajectory.shape[0]
    crossings = []

    # Find threshold crossings
    for i in range(n_steps - 1):
        curr_val = trajectory[i, coord_idx]
        next_val = trajectory[i + 1, coord_idx]

        # Check for crossing
        if (curr_val <= threshold < next_val) or (curr_val >= threshold > next_val):
            # Extract bit from crossing point
            crossing_coords = 0.5 * (trajectory[i] + trajectory[i + 1])

            # Use sum of coordinates modulo 2 as bit
            coord_sum = np.sum(crossing_coords)
            bit = int(coord_sum * 1000) & 1
            crossings.append(bit)

    return np.array(crossings, dtype=np.uint8)


@numba.njit(cache=True)
def _poincare_extract_hyperplane_jit(
    trajectory: np.ndarray, normal: np.ndarray, offset: float
) -> np.ndarray:
    """
    JIT-compiled Poincaré extraction using hyperplane crossings.
    """
    n_steps = trajectory.shape[0]
    crossings = []

    # Normalize the normal vector
    normal_normalized = normal / np.linalg.norm(normal)

    # Find hyperplane crossings
    for i in range(n_steps - 1):
        # Compute distance to hyperplane for consecutive points
        dist_curr = np.dot(trajectory[i], normal_normalized) - offset
        dist_next = np.dot(trajectory[i + 1], normal_normalized) - offset

        # Check for crossing (sign change)
        if dist_curr * dist_next < 0:
            # Interpolate crossing point
            alpha = abs(dist_curr) / (abs(dist_curr) + abs(dist_next))
            crossing_point = trajectory[i] + alpha * (trajectory[i + 1] - trajectory[i])

            # Extract bit from crossing point coordinates
            coord_sum = np.sum(crossing_point)
            bit = int(coord_sum * 1000) & 1
            crossings.append(bit)

    return np.array(crossings, dtype=np.uint8)


@numba.njit(cache=True)
def _xor_shift_postprocess_jit(bits: np.ndarray) -> np.ndarray:
    """
    Apply XOR-shift post-processing to improve bit quality.

    This simple post-processing helps break weak correlations
    in the extracted bit stream.
    """
    n_bits = len(bits)
    if n_bits < 8:
        return bits

    processed = np.zeros(n_bits - 7, dtype=np.uint8)

    for i in range(n_bits - 7):
        # XOR with bits at different offsets
        xor_val = bits[i] ^ bits[i + 1] ^ bits[i + 3] ^ bits[i + 7]
        processed[i] = xor_val

    return processed


@numba.njit(cache=True)
def _von_neumann_debias_jit(bits: np.ndarray) -> np.ndarray:
    """
    Apply von Neumann debiasing to remove statistical bias.

    This method looks at consecutive bit pairs and only keeps
    bits from pairs (0,1) -> 0 and (1,0) -> 1, discarding
    pairs (0,0) and (1,1).
    """
    # Ensure a consistent unsigned type regardless of input
    bits = bits.astype(np.uint8)

    n_bits = len(bits)
    if n_bits < 2:
        return bits

    debiased = []

    i = 0
    while i < n_bits - 1:
        bit1, bit2 = bits[i], bits[i + 1]

        if bit1 == 0 and bit2 == 1:
            debiased.append(0)
            i += 2
        elif bit1 == 1 and bit2 == 0:
            debiased.append(1)
            i += 2
        else:
            # Skip this pair
            i += 2

    return np.array(debiased, dtype=np.uint8)


class EntropyAnalyzer:
    """
    Analyzer for measuring entropy quality of extracted bits.

    This class provides various metrics to assess the quality
    of the random bit stream generated by extraction methods.
    """

    def __init__(self):
        """Initialize entropy analyzer."""
        self._analyzer = StatisticalAnalyzer()

    def shannon_entropy(self, bits: np.ndarray) -> float:
        """
        Calculate Shannon entropy of bit sequence.

        Parameters
        ----------
        bits : ndarray
            Binary sequence

        Returns
        -------
        float
            Shannon entropy in bits
        """
        if len(bits) == 0:
            return 0.0

        # Count 0s and 1s
        counts = np.bincount(bits, minlength=2)
        probabilities = counts / len(bits)

        # Calculate entropy
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * np.log2(p)

        return entropy

    def autocorrelation(
        self, bits: np.ndarray, max_lag: int = 100
    ) -> dict[str, np.ndarray | float | int]:
        """
        Calculate autocorrelation statistics for bit sequence.

        Returns a dictionary with autocorrelation values and independence metrics.
        """
        return self._analyzer.autocorrelation_test(bits, max_lag)

    def frequency_test(self, bits: np.ndarray) -> dict[str, float | int | bool]:
        """
        Perform frequency/uniformity test on the bit sequence.

        Delegates to the comprehensive StatisticalAnalyzer implementation to
        provide chi-square, p-value, and uniformity assessment.
        """
        result = self._analyzer.frequency_analysis(bits)
        if "uniform" in result:
            result["uniform"] = bool(result["uniform"])
        return result

    def spectral_test(self, bits: np.ndarray) -> dict[str, np.ndarray | float | int]:
        """Perform spectral analysis of the sequence."""
        return self._analyzer.spectral_test(bits)

    def entropy_measures(
        self, bits: np.ndarray, block_size: int = 8
    ) -> dict[str, float]:
        """Compute Shannon and block entropy measures."""
        return self._analyzer.entropy_measures(bits, block_size)

    def comprehensive_analysis(self, bits: np.ndarray) -> dict[str, object]:
        """
        Run a suite of statistical tests and quality metrics on the sequence.

        This aggregates frequency, runs, autocorrelation, spectral, and entropy
        analyses and provides an overall quality summary.
        """
        return self._analyzer.comprehensive_analysis(bits)
