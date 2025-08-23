"""
Statistical analysis tools for chaos-based random number generators.

This module provides comprehensive statistical analysis capabilities
including Lyapunov exponent calculation, phase space analysis, and
various randomness quality metrics.
"""

from typing import Tuple, List, Optional, Union, Dict, Any
import numpy as np
import numba
from scipy import signal, stats
from scipy.fft import fft, fftfreq


class LyapunovCalculator:
    """
    Advanced Lyapunov exponent calculator for chaotic systems.
    
    This class provides multiple methods for calculating Lyapunov exponents,
    which quantify the rate of divergence of nearby trajectories in phase space.
    """
    
    def __init__(self, method: str = 'wolf', embedding_dim: int = 3):
        """
        Initialize Lyapunov calculator.
        
        Parameters
        ----------
        method : str, default='wolf'
            Calculation method: 'wolf', 'rosenstein', or 'kantz'
        embedding_dim : int, default=3
            Embedding dimension for time delay embedding
        """
        self.method = method
        self.embedding_dim = embedding_dim
    
    def calculate_largest_exponent(
        self, 
        time_series: np.ndarray,
        dt: float = 1.0,
        min_tsep: int = None,
        max_tsep: int = None
    ) -> float:
        """
        Calculate the largest Lyapunov exponent from time series data.
        
        Parameters
        ----------
        time_series : ndarray
            1D time series data
        dt : float, default=1.0
            Time step of the series
        min_tsep : int, optional
            Minimum temporal separation for nearest neighbors
        max_tsep : int, optional
            Maximum temporal separation for analysis
            
        Returns
        -------
        float
            Largest Lyapunov exponent
        """
        if self.method == 'wolf':
            return self._wolf_method(time_series, dt, min_tsep, max_tsep)
        elif self.method == 'rosenstein':
            return self._rosenstein_method(time_series, dt, min_tsep)
        elif self.method == 'kantz':
            return self._kantz_method(time_series, dt, min_tsep)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _wolf_method(
        self, 
        time_series: np.ndarray, 
        dt: float,
        min_tsep: Optional[int],
        max_tsep: Optional[int]
    ) -> float:
        """Wolf et al. method for Lyapunov exponent calculation."""
        # Time delay embedding
        embedded = self._time_delay_embedding(time_series)
        
        n_points = len(embedded)
        if min_tsep is None:
            min_tsep = int(0.01 * n_points)
        if max_tsep is None:
            max_tsep = int(0.1 * n_points)
        
        # Initialize
        sum_log_div = 0.0
        n_pairs = 0
        
        for i in range(n_points - max_tsep):
            # Find nearest neighbor with sufficient temporal separation
            distances = []
            indices = []
            
            for j in range(n_points):
                if abs(i - j) >= min_tsep:
                    dist = np.linalg.norm(embedded[i] - embedded[j])
                    distances.append(dist)
                    indices.append(j)
            
            if not distances:
                continue
            
            # Find nearest neighbor
            min_idx = np.argmin(distances)
            nn_idx = indices[min_idx]
            
            # Check if we can follow the trajectory
            if nn_idx + max_tsep >= n_points:
                continue
            
            # Calculate divergence
            initial_dist = distances[min_idx]
            if initial_dist == 0:
                continue
            
            final_dist = np.linalg.norm(
                embedded[i + max_tsep] - embedded[nn_idx + max_tsep]
            )
            
            if final_dist > 0:
                sum_log_div += np.log(final_dist / initial_dist)
                n_pairs += 1
        
        if n_pairs == 0:
            return 0.0
        
        return sum_log_div / (n_pairs * max_tsep * dt)
    
    def _rosenstein_method(
        self, 
        time_series: np.ndarray, 
        dt: float, 
        min_tsep: Optional[int]
    ) -> float:
        """Rosenstein et al. method for Lyapunov exponent calculation."""
        embedded = self._time_delay_embedding(time_series)
        n_points = len(embedded)
        
        if min_tsep is None:
            min_tsep = int(0.01 * n_points)
        
        max_iter = min(int(0.1 * n_points), n_points // 4)
        log_divergence = np.zeros(max_iter)
        
        # For each point, find nearest neighbor and track divergence
        for i in range(n_points - max_iter):
            distances = []
            indices = []
            
            for j in range(n_points - max_iter):
                if abs(i - j) >= min_tsep:
                    dist = np.linalg.norm(embedded[i] - embedded[j])
                    distances.append(dist)
                    indices.append(j)
            
            if not distances:
                continue
            
            min_idx = np.argmin(distances)
            nn_idx = indices[min_idx]
            initial_dist = distances[min_idx]
            
            if initial_dist == 0:
                continue
            
            # Track divergence over time
            for k in range(max_iter):
                if i + k >= n_points or nn_idx + k >= n_points:
                    break
                
                current_dist = np.linalg.norm(
                    embedded[i + k] - embedded[nn_idx + k]
                )
                
                if current_dist > 0:
                    log_divergence[k] += np.log(current_dist / initial_dist)
        
        # Linear fit to get Lyapunov exponent
        time_steps = np.arange(max_iter) * dt
        valid_points = log_divergence != 0
        
        if np.sum(valid_points) < 2:
            return 0.0
        
        slope, _, _, _, _ = stats.linregress(
            time_steps[valid_points], 
            log_divergence[valid_points]
        )
        
        return slope
    
    def _kantz_method(
        self, 
        time_series: np.ndarray, 
        dt: float, 
        min_tsep: Optional[int]
    ) -> float:
        """Kantz method for Lyapunov exponent calculation."""
        # This is a simplified version of the Kantz method
        return self._rosenstein_method(time_series, dt, min_tsep)
    
    def _time_delay_embedding(self, time_series: np.ndarray, tau: int = 1) -> np.ndarray:
        """
        Perform time delay embedding of a time series.
        
        Parameters
        ----------
        time_series : ndarray
            1D time series
        tau : int, default=1
            Time delay
            
        Returns
        -------
        ndarray
            Embedded time series [n_points, embedding_dim]
        """
        n = len(time_series)
        n_vectors = n - (self.embedding_dim - 1) * tau
        
        if n_vectors <= 0:
            raise ValueError("Time series too short for embedding")
        
        embedded = np.zeros((n_vectors, self.embedding_dim))
        
        for i in range(self.embedding_dim):
            start_idx = i * tau
            end_idx = start_idx + n_vectors
            embedded[:, i] = time_series[start_idx:end_idx]
        
        return embedded


class PhaseSpaceAnalyzer:
    """
    Phase space analysis tools for chaotic systems.
    
    This class provides methods for analyzing the geometry and
    dynamics of phase space trajectories.
    """
    
    def __init__(self):
        """Initialize phase space analyzer."""
        pass
    
    def calculate_correlation_dimension(
        self, 
        trajectory: np.ndarray,
        r_min: float = 1e-6,
        r_max: float = 1.0,
        n_points: int = 50
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Calculate correlation dimension using Grassberger-Procaccia algorithm.
        
        Parameters
        ----------
        trajectory : ndarray
            Phase space trajectory [n_steps, n_dims]
        r_min : float, default=1e-6
            Minimum radius for analysis
        r_max : float, default=1.0
            Maximum radius for analysis
        n_points : int, default=50
            Number of radius values to test
            
        Returns
        -------
        correlation_dimension : float
            Estimated correlation dimension
        radii : ndarray
            Radius values used
        correlations : ndarray
            Correlation integrals
        """
        n_traj = len(trajectory)
        radii = np.logspace(np.log10(r_min), np.log10(r_max), n_points)
        correlations = np.zeros(n_points)
        
        # Calculate pairwise distances
        distances = self._pairwise_distances(trajectory)
        
        # Calculate correlation integral for each radius
        for i, r in enumerate(radii):
            count = np.sum(distances < r)
            # Subtract diagonal elements (self-distances)
            count -= n_traj
            correlations[i] = count / (n_traj * (n_traj - 1))
        
        # Estimate dimension from slope of log-log plot
        log_r = np.log10(radii)
        log_c = np.log10(correlations + 1e-15)  # Avoid log(0)
        
        # Find linear region (middle portion)
        start_idx = n_points // 4
        end_idx = 3 * n_points // 4
        
        slope, _, _, _, _ = stats.linregress(
            log_r[start_idx:end_idx], 
            log_c[start_idx:end_idx]
        )
        
        return slope, radii, correlations
    
    def _pairwise_distances(self, trajectory: np.ndarray) -> np.ndarray:
        """Calculate pairwise distances between trajectory points."""
        n_points = len(trajectory)
        distances = np.zeros((n_points, n_points))
        
        for i in range(n_points):
            for j in range(i + 1, n_points):
                dist = np.linalg.norm(trajectory[i] - trajectory[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances
    
    def calculate_recurrence_plot(
        self, 
        trajectory: np.ndarray,
        threshold: float = 0.1
    ) -> np.ndarray:
        """
        Calculate recurrence plot for trajectory analysis.
        
        Parameters
        ----------
        trajectory : ndarray
            Phase space trajectory
        threshold : float, default=0.1
            Recurrence threshold
            
        Returns
        -------
        ndarray
            Recurrence plot matrix
        """
        distances = self._pairwise_distances(trajectory)
        return (distances < threshold).astype(int)
    
    def poincare_section(
        self, 
        trajectory: np.ndarray,
        plane_normal: np.ndarray,
        plane_offset: float = 0.0
    ) -> np.ndarray:
        """
        Calculate Poincaré section of trajectory.
        
        Parameters
        ----------
        trajectory : ndarray
            Phase space trajectory [n_steps, n_dims]
        plane_normal : ndarray
            Normal vector to section plane
        plane_offset : float, default=0.0
            Plane offset from origin
            
        Returns
        -------
        ndarray
            Points where trajectory crosses the section
        """
        n_steps = len(trajectory)
        crossings = []
        
        # Normalize plane normal
        plane_normal = plane_normal / np.linalg.norm(plane_normal)
        
        for i in range(n_steps - 1):
            # Calculate signed distances to plane
            dist1 = np.dot(trajectory[i], plane_normal) - plane_offset
            dist2 = np.dot(trajectory[i + 1], plane_normal) - plane_offset
            
            # Check for crossing (sign change)
            if dist1 * dist2 < 0:
                # Interpolate crossing point
                alpha = abs(dist1) / (abs(dist1) + abs(dist2))
                crossing = trajectory[i] + alpha * (trajectory[i + 1] - trajectory[i])
                crossings.append(crossing)
        
        return np.array(crossings) if crossings else np.array([]).reshape(0, trajectory.shape[1])


class StatisticalAnalyzer:
    """
    Comprehensive statistical analyzer for random sequences.
    
    This class provides various statistical tests and measures
    for assessing the quality of random number sequences.
    """
    
    def __init__(self):
        """Initialize statistical analyzer."""
        pass
    
    def frequency_analysis(self, sequence: np.ndarray) -> Dict[str, float]:
        """
        Perform frequency analysis on binary sequence.
        
        Parameters
        ----------
        sequence : ndarray
            Binary sequence (0s and 1s)
            
        Returns
        -------
        dict
            Frequency analysis results
        """
        n = len(sequence)
        if n == 0:
            return {}
        
        # Basic frequency statistics
        ones = np.sum(sequence)
        zeros = n - ones
        
        # Chi-square test for uniformity
        expected = n / 2
        chi_square = ((ones - expected)**2 + (zeros - expected)**2) / expected
        
        # Approximate p-value
        p_value = 1 - stats.chi2.cdf(chi_square, df=1)
        
        return {
            'ones_count': int(ones),
            'zeros_count': int(zeros),
            'ones_frequency': ones / n,
            'chi_square': chi_square,
            'p_value': p_value,
            'uniform': p_value > 0.01
        }
    
    def runs_test(self, sequence: np.ndarray) -> Dict[str, float]:
        """
        Perform runs test on binary sequence.
        
        Parameters
        ----------
        sequence : ndarray
            Binary sequence
            
        Returns
        -------
        dict
            Runs test results
        """
        n = len(sequence)
        if n <= 1:
            return {}
        
        # Count runs
        runs = 1
        for i in range(1, n):
            if sequence[i] != sequence[i-1]:
                runs += 1
        
        # Expected runs and variance
        ones = np.sum(sequence)
        zeros = n - ones
        
        if ones == 0 or zeros == 0:
            return {'runs': runs, 'expected_runs': n, 'p_value': 0.0}
        
        expected_runs = (2 * ones * zeros) / n + 1
        variance = (2 * ones * zeros * (2 * ones * zeros - n)) / (n**2 * (n - 1))
        
        if variance <= 0:
            return {'runs': runs, 'expected_runs': expected_runs, 'p_value': 1.0}
        
        # Test statistic
        z = (runs - expected_runs) / np.sqrt(variance)
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        return {
            'runs': runs,
            'expected_runs': expected_runs,
            'z_statistic': z,
            'p_value': p_value,
            'random': p_value > 0.01
        }
    
    def autocorrelation_test(
        self, 
        sequence: np.ndarray, 
        max_lag: int = 100
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Calculate autocorrelation function of sequence.
        
        Parameters
        ----------
        sequence : ndarray
            Input sequence
        max_lag : int, default=100
            Maximum lag to compute
            
        Returns
        -------
        dict
            Autocorrelation analysis results
        """
        n = len(sequence)
        if n <= max_lag:
            max_lag = n - 1
        
        # Convert to centered values (-1, +1)
        centered = 2 * sequence.astype(float) - 1
        
        # Calculate autocorrelation
        autocorr = np.correlate(centered, centered, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Limit to max_lag
        autocorr = autocorr[:max_lag + 1]
        
        # Statistical significance test
        # Under null hypothesis, autocorrelations should be ~N(0, 1/n)
        std_error = 1.0 / np.sqrt(n)
        significant_lags = np.abs(autocorr[1:]) > 2 * std_error
        
        return {
            'autocorrelation': autocorr,
            'max_autocorr': np.max(np.abs(autocorr[1:])),
            'significant_lags': np.sum(significant_lags),
            'independence': np.sum(significant_lags) / len(significant_lags) < 0.05
        }
    
    def spectral_test(self, sequence: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
        """
        Perform spectral analysis of sequence.
        
        Parameters
        ----------
        sequence : ndarray
            Input sequence
            
        Returns
        -------
        dict
            Spectral analysis results
        """
        n = len(sequence)
        if n < 2:
            return {}
        
        # Convert to bipolar (-1, +1)
        bipolar = 2 * sequence.astype(float) - 1
        
        # Calculate power spectral density
        frequencies, psd = signal.periodogram(bipolar)
        
        # For random sequence, PSD should be approximately flat
        # Test for significant peaks
        mean_psd = np.mean(psd[1:])  # Exclude DC component
        std_psd = np.std(psd[1:])
        
        # Find peaks above 3 sigma
        threshold = mean_psd + 3 * std_psd
        significant_peaks = np.sum(psd[1:] > threshold)
        
        return {
            'frequencies': frequencies,
            'psd': psd,
            'mean_psd': mean_psd,
            'psd_variance': std_psd**2,
            'significant_peaks': significant_peaks,
            'spectral_flatness': mean_psd / (np.max(psd[1:]) + 1e-15)
        }
    
    def entropy_measures(self, sequence: np.ndarray, block_size: int = 8) -> Dict[str, float]:
        """
        Calculate various entropy measures.
        
        Parameters
        ----------
        sequence : ndarray
            Input sequence
        block_size : int, default=8
            Block size for block entropy calculation
            
        Returns
        -------
        dict
            Entropy measures
        """
        n = len(sequence)
        
        # Shannon entropy (per bit)
        if n > 0:
            p0 = np.sum(sequence == 0) / n
            p1 = np.sum(sequence == 1) / n
            
            shannon = 0.0
            if p0 > 0:
                shannon -= p0 * np.log2(p0)
            if p1 > 0:
                shannon -= p1 * np.log2(p1)
        else:
            shannon = 0.0
        
        # Block entropy
        if n >= block_size:
            n_blocks = n - block_size + 1
            blocks = {}
            
            for i in range(n_blocks):
                block = tuple(sequence[i:i + block_size])
                blocks[block] = blocks.get(block, 0) + 1
            
            block_entropy = 0.0
            for count in blocks.values():
                p = count / n_blocks
                if p > 0:
                    block_entropy -= p * np.log2(p)
            
            # Normalize by block size
            block_entropy /= block_size
        else:
            block_entropy = 0.0
        
        return {
            'shannon_entropy': shannon,
            'block_entropy': block_entropy,
            'entropy_per_block': block_entropy * block_size
        }
    
    def comprehensive_analysis(self, sequence: np.ndarray) -> Dict[str, Any]:
        """
        Perform comprehensive statistical analysis.
        
        Parameters
        ----------
        sequence : ndarray
            Binary sequence to analyze
            
        Returns
        -------
        dict
            Complete analysis results
        """
        results = {
            'sequence_length': len(sequence),
            'frequency': self.frequency_analysis(sequence),
            'runs': self.runs_test(sequence),
            'autocorrelation': self.autocorrelation_test(sequence),
            'spectral': self.spectral_test(sequence),
            'entropy': self.entropy_measures(sequence)
        }
        
        # Overall quality assessment
        tests_passed = 0
        total_tests = 0
        
        if 'uniform' in results['frequency']:
            total_tests += 1
            if results['frequency']['uniform']:
                tests_passed += 1
        
        if 'random' in results['runs']:
            total_tests += 1
            if results['runs']['random']:
                tests_passed += 1
        
        if 'independence' in results['autocorrelation']:
            total_tests += 1
            if results['autocorrelation']['independence']:
                tests_passed += 1
        
        results['overall_quality'] = {
            'tests_passed': tests_passed,
            'total_tests': total_tests,
            'pass_rate': tests_passed / total_tests if total_tests > 0 else 0.0
        }
        
        return results