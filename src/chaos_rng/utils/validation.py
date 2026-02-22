"""
Validation utilities for chaos-based random number generators.

This module provides comprehensive validation tools including NIST SP 800-22
statistical tests, entropy validation, and continuous monitoring capabilities.
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

try:
    import nistrng

    NIST_AVAILABLE = True
except ImportError:
    NIST_AVAILABLE = False


@dataclass
class TestResult:
    """Container for statistical test results."""

    test_name: str
    statistic: float
    p_value: float
    passed: bool
    threshold: float = 0.01
    additional_info: Optional[dict[str, Any]] = None


class NISTTestSuite:
    """
    NIST SP 800-22 statistical test suite implementation.

    This class provides access to the NIST Statistical Test Suite
    for random and pseudorandom number generators for cryptographic applications.
    """

    def __init__(self, significance_level: float = 0.01):
        """
        Initialize NIST test suite.

        Parameters
        ----------
        significance_level : float, default=0.01
            Significance level for statistical tests
        """
        self.significance_level = significance_level

        if not NIST_AVAILABLE:
            raise ImportError(
                "nistrng package is required for NIST tests. "
                "Install with: pip install nistrng"
            )

    def run_all_tests(self, binary_sequence: np.ndarray) -> dict[str, TestResult]:
        """
        Run all NIST tests on binary sequence.

        Parameters
        ----------
        binary_sequence : ndarray
            Binary sequence (0s and 1s) to test

        Returns
        -------
        dict
            Dictionary of test results
        """
        if not NIST_AVAILABLE:
            raise RuntimeError("NIST tests not available")

        # Convert to string format required by nistrng
        bit_string = "".join(binary_sequence.astype(str))

        results = {}

        # List of NIST tests to run
        test_methods = [
            ("frequency", self._frequency_test),
            ("block_frequency", self._block_frequency_test),
            ("runs", self._runs_test),
            ("longest_run", self._longest_run_test),
            ("binary_matrix_rank", self._binary_matrix_rank_test),
            ("dft", self._dft_test),
            ("non_overlapping_template", self._non_overlapping_template_test),
            ("overlapping_template", self._overlapping_template_test),
            ("universal", self._universal_test),
            ("linear_complexity", self._linear_complexity_test),
            ("serial", self._serial_test),
            ("approximate_entropy", self._approximate_entropy_test),
            ("cumulative_sums", self._cumulative_sums_test),
            ("random_excursions", self._random_excursions_test),
            ("random_excursions_variant", self._random_excursions_variant_test),
        ]

        for test_name, test_method in test_methods:
            try:
                result = test_method(bit_string)
                results[test_name] = result
            except Exception as e:
                # Some tests may fail due to sequence length or other constraints
                results[test_name] = TestResult(
                    test_name=test_name,
                    statistic=0.0,
                    p_value=0.0,
                    passed=False,
                    additional_info={"error": str(e)},
                )

        return results

    def _frequency_test(self, bit_string: str) -> TestResult:
        """NIST Frequency (Monobit) Test."""
        result = nistrng.frequency_test(bit_string)
        return TestResult(
            test_name="frequency",
            statistic=result[0],
            p_value=result[1],
            passed=result[1] >= self.significance_level,
        )

    def _block_frequency_test(self, bit_string: str) -> TestResult:
        """NIST Block Frequency Test."""
        result = nistrng.block_frequency_test(bit_string)
        return TestResult(
            test_name="block_frequency",
            statistic=result[0],
            p_value=result[1],
            passed=result[1] >= self.significance_level,
        )

    def _runs_test(self, bit_string: str) -> TestResult:
        """NIST Runs Test."""
        result = nistrng.runs_test(bit_string)
        return TestResult(
            test_name="runs",
            statistic=result[0],
            p_value=result[1],
            passed=result[1] >= self.significance_level,
        )

    def _longest_run_test(self, bit_string: str) -> TestResult:
        """NIST Test for the Longest Run of Ones in a Block."""
        result = nistrng.longest_run_ones_in_a_block_test(bit_string)
        return TestResult(
            test_name="longest_run",
            statistic=result[0],
            p_value=result[1],
            passed=result[1] >= self.significance_level,
        )

    def _binary_matrix_rank_test(self, bit_string: str) -> TestResult:
        """NIST Binary Matrix Rank Test."""
        result = nistrng.binary_matrix_rank_test(bit_string)
        return TestResult(
            test_name="binary_matrix_rank",
            statistic=result[0],
            p_value=result[1],
            passed=result[1] >= self.significance_level,
        )

    def _dft_test(self, bit_string: str) -> TestResult:
        """NIST Discrete Fourier Transform (Spectral) Test."""
        result = nistrng.dft_test(bit_string)
        return TestResult(
            test_name="dft",
            statistic=result[0],
            p_value=result[1],
            passed=result[1] >= self.significance_level,
        )

    def _non_overlapping_template_test(self, bit_string: str) -> TestResult:
        """NIST Non-overlapping Template Matching Test."""
        # Use a common template
        template = "000000001"
        result = nistrng.non_overlapping_template_matching_test(bit_string, template)
        return TestResult(
            test_name="non_overlapping_template",
            statistic=result[0],
            p_value=result[1],
            passed=result[1] >= self.significance_level,
        )

    def _overlapping_template_test(self, bit_string: str) -> TestResult:
        """NIST Overlapping Template Matching Test."""
        result = nistrng.overlapping_template_matching_test(bit_string)
        return TestResult(
            test_name="overlapping_template",
            statistic=result[0],
            p_value=result[1],
            passed=result[1] >= self.significance_level,
        )

    def _universal_test(self, bit_string: str) -> TestResult:
        """NIST Maurer's Universal Statistical Test."""
        result = nistrng.maurers_universal_test(bit_string)
        return TestResult(
            test_name="universal",
            statistic=result[0],
            p_value=result[1],
            passed=result[1] >= self.significance_level,
        )

    def _linear_complexity_test(self, bit_string: str) -> TestResult:
        """NIST Linear Complexity Test."""
        result = nistrng.linear_complexity_test(bit_string)
        return TestResult(
            test_name="linear_complexity",
            statistic=result[0],
            p_value=result[1],
            passed=result[1] >= self.significance_level,
        )

    def _serial_test(self, bit_string: str) -> TestResult:
        """NIST Serial Test."""
        result = nistrng.serial_test(bit_string)
        # Serial test returns two p-values, use the first one
        p_value = result[1][0] if isinstance(result[1], (list, tuple)) else result[1]
        return TestResult(
            test_name="serial",
            statistic=result[0],
            p_value=p_value,
            passed=p_value >= self.significance_level,
        )

    def _approximate_entropy_test(self, bit_string: str) -> TestResult:
        """NIST Approximate Entropy Test."""
        result = nistrng.approximate_entropy_test(bit_string)
        return TestResult(
            test_name="approximate_entropy",
            statistic=result[0],
            p_value=result[1],
            passed=result[1] >= self.significance_level,
        )

    def _cumulative_sums_test(self, bit_string: str) -> TestResult:
        """NIST Cumulative Sums Test."""
        result = nistrng.cumulative_sums_test(bit_string)
        # Test returns two p-values (forward and reverse), use minimum
        p_values = result[1] if isinstance(result[1], (list, tuple)) else [result[1]]
        min_p_value = min(p_values)
        return TestResult(
            test_name="cumulative_sums",
            statistic=result[0],
            p_value=min_p_value,
            passed=min_p_value >= self.significance_level,
        )

    def _random_excursions_test(self, bit_string: str) -> TestResult:
        """NIST Random Excursions Test."""
        try:
            result = nistrng.random_excursions_test(bit_string)
            # This test may return multiple p-values, use minimum
            p_values = (
                result[1] if isinstance(result[1], (list, tuple)) else [result[1]]
            )
            min_p_value = min(p_values) if p_values else 0.0
            return TestResult(
                test_name="random_excursions",
                statistic=result[0] if result[0] else 0.0,
                p_value=min_p_value,
                passed=min_p_value >= self.significance_level,
            )
        except Exception:
            # Test may fail for sequences that don't meet prerequisites
            return TestResult(
                test_name="random_excursions",
                statistic=0.0,
                p_value=0.0,
                passed=False,
                additional_info={"error": "Test prerequisites not met"},
            )

    def _random_excursions_variant_test(self, bit_string: str) -> TestResult:
        """NIST Random Excursions Variant Test."""
        try:
            result = nistrng.random_excursions_variant_test(bit_string)
            # This test may return multiple p-values, use minimum
            p_values = (
                result[1] if isinstance(result[1], (list, tuple)) else [result[1]]
            )
            min_p_value = min(p_values) if p_values else 0.0
            return TestResult(
                test_name="random_excursions_variant",
                statistic=result[0] if result[0] else 0.0,
                p_value=min_p_value,
                passed=min_p_value >= self.significance_level,
            )
        except Exception:
            return TestResult(
                test_name="random_excursions_variant",
                statistic=0.0,
                p_value=0.0,
                passed=False,
                additional_info={"error": "Test prerequisites not met"},
            )


class EntropyValidator:
    """
    Entropy validation and measurement utilities.

    This class provides methods for measuring and validating
    the entropy content of random sequences.
    """

    def __init__(self, min_entropy_rate: float = 0.99):
        """
        Initialize entropy validator.

        Parameters
        ----------
        min_entropy_rate : float, default=0.99
            Minimum acceptable entropy rate (bits per bit)
        """
        self.min_entropy_rate = min_entropy_rate

    def estimate_entropy(
        self, sequence: np.ndarray, method: str = "shannon"
    ) -> dict[str, float]:
        """
        Estimate entropy of a sequence using various methods.

        Parameters
        ----------
        sequence : ndarray
            Input sequence
        method : str, default='shannon'
            Entropy estimation method: 'shannon', 'min', 'collision', or 'all'

        Returns
        -------
        dict
            Entropy estimates
        """
        results = {}

        if method in ["shannon", "all"]:
            results["shannon"] = self._shannon_entropy(sequence)

        if method in ["min", "all"]:
            results["min_entropy"] = self._min_entropy(sequence)

        if method in ["collision", "all"]:
            results["collision_entropy"] = self._collision_entropy(sequence)

        return results

    def _shannon_entropy(self, sequence: np.ndarray) -> float:
        """Calculate Shannon entropy per bit."""
        if len(sequence) == 0:
            return 0.0

        # Count symbol frequencies
        unique, counts = np.unique(sequence, return_counts=True)
        probabilities = counts / len(sequence)

        # Calculate Shannon entropy
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-15))

        # Normalize by log2 of alphabet size
        alphabet_size = len(unique)
        if alphabet_size > 1:
            entropy /= np.log2(alphabet_size)

        return entropy

    def _min_entropy(self, sequence: np.ndarray) -> float:
        """Calculate min-entropy (worst-case entropy)."""
        if len(sequence) == 0:
            return 0.0

        # Find most frequent symbol
        unique, counts = np.unique(sequence, return_counts=True)
        max_prob = np.max(counts) / len(sequence)

        # Min-entropy is -log2 of maximum probability
        return -np.log2(max_prob + 1e-15)

    def _collision_entropy(self, sequence: np.ndarray) -> float:
        """Calculate collision entropy."""
        if len(sequence) <= 1:
            return 0.0

        # Count symbol frequencies
        unique, counts = np.unique(sequence, return_counts=True)

        # Calculate collision probability
        collision_prob = np.sum((counts / len(sequence)) ** 2)

        # Collision entropy
        return -np.log2(collision_prob + 1e-15)

    def validate_entropy_rate(self, sequence: np.ndarray) -> dict[str, Any]:
        """
        Validate entropy rate of sequence.

        Parameters
        ----------
        sequence : ndarray
            Binary sequence to validate

        Returns
        -------
        dict
            Validation results
        """
        # Calculate various entropy measures
        entropies = self.estimate_entropy(sequence, method="all")

        # Check if entropy rates meet minimum requirements
        validation_results = {}

        for entropy_type, entropy_value in entropies.items():
            validation_results[f"{entropy_type}_rate"] = entropy_value
            validation_results[f"{entropy_type}_valid"] = (
                entropy_value >= self.min_entropy_rate
            )

        # Overall validation
        all_valid = all(validation_results[f"{et}_valid"] for et in entropies.keys())

        validation_results["overall_valid"] = all_valid
        validation_results["min_required_rate"] = self.min_entropy_rate

        return validation_results


class ContinuousValidator:
    """
    Continuous validation monitor for chaos-based RNG.

    This class provides real-time monitoring and validation
    of RNG output quality during operation.
    """

    def __init__(
        self,
        window_size: int = 100000,
        test_interval: int = 10000,
        alert_threshold: int = 3,
    ):
        """
        Initialize continuous validator.

        Parameters
        ----------
        window_size : int, default=100000
            Size of sliding window for analysis
        test_interval : int, default=10000
            Interval between statistical tests
        alert_threshold : int, default=3
            Number of consecutive failures before alert
        """
        self.window_size = window_size
        self.test_interval = test_interval
        self.alert_threshold = alert_threshold

        # Buffers and counters
        self.buffer = np.array([], dtype=np.uint8)
        self.test_counter = 0
        self.failure_count = 0
        self.total_tests = 0
        self.passed_tests = 0

        # Test history
        self.test_history = []

        # Initialize validators
        self.entropy_validator = EntropyValidator()

        # Simple statistical tests (NIST tests are expensive for continuous monitoring)
        self.quick_tests = [
            self._frequency_test,
            self._runs_test,
            self._autocorrelation_test,
        ]

    def add_data(self, new_data: np.ndarray) -> dict[str, Any]:
        """
        Add new data and perform validation if needed.

        Parameters
        ----------
        new_data : ndarray
            New binary data to add

        Returns
        -------
        dict
            Validation results if tests were performed, None otherwise
        """
        # Add to buffer
        self.buffer = np.concatenate([self.buffer, new_data.astype(np.uint8)])

        # Maintain window size
        if len(self.buffer) > self.window_size:
            excess = len(self.buffer) - self.window_size
            self.buffer = self.buffer[excess:]

        self.test_counter += len(new_data)

        # Check if it's time to run tests
        if self.test_counter >= self.test_interval and len(self.buffer) >= 1000:
            results = self._run_quick_tests()
            self.test_counter = 0
            self._update_statistics(results)
            return results

        return None

    def _run_quick_tests(self) -> dict[str, Any]:
        """Run quick statistical tests on current buffer."""
        if len(self.buffer) == 0:
            return {}

        test_results = {}
        passed_count = 0

        # Run each quick test
        for test_func in self.quick_tests:
            try:
                result = test_func(self.buffer)
                test_results[result.test_name] = result
                if result.passed:
                    passed_count += 1
            except Exception as e:
                test_results[f"{test_func.__name__}_error"] = str(e)

        # Entropy validation
        try:
            entropy_results = self.entropy_validator.validate_entropy_rate(self.buffer)
            test_results["entropy"] = entropy_results
            if entropy_results.get("overall_valid", False):
                passed_count += 1
        except Exception as e:
            test_results["entropy_error"] = str(e)

        # Overall assessment
        total_tests = len(self.quick_tests) + 1  # +1 for entropy
        test_results["summary"] = {
            "passed_tests": passed_count,
            "total_tests": total_tests,
            "pass_rate": passed_count / total_tests,
            "timestamp": __import__("time").time(),
        }

        return test_results

    def _frequency_test(self, sequence: np.ndarray) -> TestResult:
        """Quick frequency test."""
        n = len(sequence)
        ones = np.sum(sequence)

        # Chi-square test
        expected = n / 2
        chi_square = ((ones - expected) ** 2 + (n - ones - expected) ** 2) / expected

        # Approximate p-value
        from scipy.stats import chi2

        p_value = 1 - chi2.cdf(chi_square, df=1)

        return TestResult(
            test_name="frequency",
            statistic=chi_square,
            p_value=p_value,
            passed=p_value >= 0.01,
        )

    def _runs_test(self, sequence: np.ndarray) -> TestResult:
        """Quick runs test."""
        n = len(sequence)
        if n <= 1:
            return TestResult("runs", 0, 0, False)

        # Count runs
        runs = 1
        for i in range(1, n):
            if sequence[i] != sequence[i - 1]:
                runs += 1

        # Expected runs
        ones = np.sum(sequence)
        zeros = n - ones

        if ones == 0 or zeros == 0:
            return TestResult("runs", runs, 0, False)

        expected_runs = (2 * ones * zeros) / n + 1
        variance = (2 * ones * zeros * (2 * ones * zeros - n)) / (n**2 * (n - 1))

        if variance <= 0:
            return TestResult("runs", runs, 1, True)

        # Test statistic
        z = abs(runs - expected_runs) / np.sqrt(variance)
        from scipy.stats import norm

        p_value = 2 * (1 - norm.cdf(z))

        return TestResult(
            test_name="runs", statistic=z, p_value=p_value, passed=p_value >= 0.01
        )

    def _autocorrelation_test(self, sequence: np.ndarray) -> TestResult:
        """Quick autocorrelation test."""
        n = len(sequence)
        if n < 100:
            return TestResult("autocorrelation", 0, 1, True)

        # Convert to bipolar
        bipolar = 2 * sequence.astype(float) - 1

        # Calculate autocorrelation at lag 1
        autocorr_1 = np.mean(bipolar[:-1] * bipolar[1:])

        # Standard error under null hypothesis
        std_error = 1.0 / np.sqrt(n)

        # Test statistic
        z = abs(autocorr_1) / std_error
        from scipy.stats import norm

        p_value = 2 * (1 - norm.cdf(z))

        return TestResult(
            test_name="autocorrelation",
            statistic=z,
            p_value=p_value,
            passed=p_value >= 0.01,
        )

    def _update_statistics(self, results: dict[str, Any]) -> None:
        """Update running statistics."""
        self.total_tests += 1

        summary = results.get("summary", {})
        pass_rate = summary.get("pass_rate", 0.0)

        if pass_rate >= 0.8:  # Consider passed if 80% of tests pass
            self.passed_tests += 1
            self.failure_count = 0
        else:
            self.failure_count += 1

        # Store in history (keep last 100 results)
        self.test_history.append(results)
        if len(self.test_history) > 100:
            self.test_history.pop(0)

    def get_status(self) -> dict[str, Any]:
        """Get current validation status."""
        return {
            "buffer_size": len(self.buffer),
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "overall_pass_rate": self.passed_tests / max(1, self.total_tests),
            "consecutive_failures": self.failure_count,
            "alert_active": self.failure_count >= self.alert_threshold,
            "test_history_length": len(self.test_history),
        }

    def reset(self) -> None:
        """Reset the validator state."""
        self.buffer = np.array([], dtype=np.uint8)
        self.test_counter = 0
        self.failure_count = 0
        self.total_tests = 0
        self.passed_tests = 0
        self.test_history = []
