"""
Tests for bit extraction methods.

This module tests the various bit extraction methods used to
convert chaotic trajectories into random bit sequences.
"""

import pytest
import numpy as np
from chaos_rng.generators.bit_extractors import (
    LSBExtractor, ThresholdExtractor, PoincareExtractor,
    BitExtractor, EntropyAnalyzer
)


class TestLSBExtractor:
    """Test the LSB (Least Significant Bit) extractor."""
    
    def test_initialization(self):
        """Test LSB extractor initialization."""
        extractor = LSBExtractor()
        assert extractor.scale_factor == 2**32
        
        # Custom scale factor
        extractor_custom = LSBExtractor(scale_factor=1000)
        assert extractor_custom.scale_factor == 1000
    
    def test_extract_bits(self):
        """Test bit extraction from trajectory."""
        extractor = LSBExtractor()
        
        # Create test trajectory
        trajectory = np.random.randn(100, 6)  # 100 time steps, 6 coordinates
        
        bits = extractor.extract(trajectory)
        
        assert isinstance(bits, np.ndarray)
        assert bits.dtype == np.uint8
        assert len(bits) == 600  # 100 * 6
        assert np.all((bits == 0) | (bits == 1))
    
    def test_deterministic_output(self):
        """Test that same input gives same output."""
        extractor = LSBExtractor()
        
        trajectory = np.array([[1.234, 5.678], [9.012, 3.456]])
        
        bits1 = extractor.extract(trajectory)
        bits2 = extractor.extract(trajectory)
        
        np.testing.assert_array_equal(bits1, bits2)
    
    def test_different_scale_factors(self):
        """Test behavior with different scale factors."""
        trajectory = np.array([[0.1, 0.2], [0.3, 0.4]])
        
        extractor1 = LSBExtractor(scale_factor=10)
        extractor2 = LSBExtractor(scale_factor=100)
        
        bits1 = extractor1.extract(trajectory)
        bits2 = extractor2.extract(trajectory)
        
        # Different scale factors should generally give different results
        assert not np.array_equal(bits1, bits2)


class TestThresholdExtractor:
    """Test the threshold-based extractor."""
    
    def test_initialization(self):
        """Test threshold extractor initialization."""
        extractor = ThresholdExtractor()
        assert extractor.coordinate_indices is None
        
        # With specific coordinates
        indices = np.array([0, 2, 4])
        extractor_custom = ThresholdExtractor(coordinate_indices=indices)
        np.testing.assert_array_equal(extractor_custom.coordinate_indices, indices)
    
    def test_extract_bits(self):
        """Test threshold bit extraction."""
        extractor = ThresholdExtractor()
        
        # Create monotonic trajectory for predictable output
        trajectory = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 4.0],
            [3.0, 3.0, 2.0]
        ])
        
        bits = extractor.extract(trajectory)
        
        assert isinstance(bits, np.ndarray)
        assert bits.dtype == np.uint8
        assert len(bits) == 6  # (3-1) * 3 coordinates
        assert np.all((bits == 0) | (bits == 1))
        
        # Check specific expected values
        # Compare (1,2,3) -> (2,1,4): [1,0,1]
        # Compare (2,1,4) -> (3,3,2): [1,1,0]
        expected = np.array([1, 0, 1, 1, 1, 0])
        np.testing.assert_array_equal(bits, expected)
    
    def test_coordinate_selection(self):
        """Test extraction with specific coordinates."""
        trajectory = np.random.randn(10, 6)
        
        # Extract from all coordinates
        extractor_all = ThresholdExtractor()
        bits_all = extractor_all.extract(trajectory)
        
        # Extract from specific coordinates
        indices = np.array([0, 2, 4])
        extractor_subset = ThresholdExtractor(coordinate_indices=indices)
        bits_subset = extractor_subset.extract(trajectory)
        
        assert len(bits_all) == 9 * 6  # (10-1) * 6
        assert len(bits_subset) == 9 * 3  # (10-1) * 3


class TestPoincareExtractor:
    """Test the Poincaré section extractor."""
    
    def test_initialization(self):
        """Test Poincaré extractor initialization."""
        extractor = PoincareExtractor()
        assert extractor.section_normal is None
        assert extractor.section_offset == 0.0
        assert extractor.coordinate_idx == 0
    
    def test_coordinate_crossing(self):
        """Test coordinate-based Poincaré sections."""
        extractor = PoincareExtractor(coordinate_idx=0, section_offset=0.0)
        
        # Create trajectory that crosses x=0
        trajectory = np.array([
            [-1.0, 0.5],
            [-0.5, 0.6],
            [0.2, 0.7],   # Crosses from negative to positive
            [0.8, 0.8],
            [0.3, 0.9],
            [-0.1, 1.0],  # Crosses from positive to negative
            [-0.6, 1.1]
        ])
        
        bits = extractor.extract(trajectory)
        
        assert isinstance(bits, np.ndarray)
        assert bits.dtype == np.uint8
        assert len(bits) == 2  # Two crossings
        assert np.all((bits == 0) | (bits == 1))
    
    def test_hyperplane_crossing(self):
        """Test hyperplane-based Poincaré sections."""
        normal = np.array([1.0, 0.0])  # Plane perpendicular to x-axis
        extractor = PoincareExtractor(section_normal=normal, section_offset=0.0)
        
        # Same trajectory as above
        trajectory = np.array([
            [-1.0, 0.5],
            [-0.5, 0.6],
            [0.2, 0.7],
            [0.8, 0.8],
            [0.3, 0.9],
            [-0.1, 1.0],
            [-0.6, 1.1]
        ])
        
        bits = extractor.extract(trajectory)
        
        assert isinstance(bits, np.ndarray)
        assert bits.dtype == np.uint8
        assert len(bits) == 2  # Two crossings
    
    def test_no_crossings(self):
        """Test behavior when no crossings occur."""
        extractor = PoincareExtractor(coordinate_idx=0, section_offset=10.0)
        
        # Trajectory that doesn't cross the section
        trajectory = np.array([
            [1.0, 0.5],
            [2.0, 0.6],
            [3.0, 0.7]
        ])
        
        bits = extractor.extract(trajectory)
        
        assert len(bits) == 0


class TestBitExtractor:
    """Test the unified BitExtractor class."""
    
    def test_initialization(self):
        """Test BitExtractor initialization."""
        extractor = BitExtractor()
        assert extractor.method == 'lsb'
        assert extractor.post_process is True
        assert extractor.debias is True
    
    @pytest.mark.parametrize("method", ['lsb', 'threshold', 'poincare'])
    def test_different_methods(self, method):
        """Test different extraction methods."""
        extractor = BitExtractor(method=method)
        
        # Create test trajectory
        trajectory = np.random.randn(100, 6)
        
        bits = extractor.extract_bits(trajectory)
        
        assert isinstance(bits, np.ndarray)
        assert bits.dtype == np.uint8
        assert len(bits) > 0
        assert np.all((bits == 0) | (bits == 1))
    
    def test_combined_method(self):
        """Test combined extraction method."""
        extractor = BitExtractor(method='combined')
        
        trajectory = np.random.randn(100, 6)
        bits = extractor.extract_bits(trajectory)
        
        assert isinstance(bits, np.ndarray)
        assert bits.dtype == np.uint8
        assert np.all((bits == 0) | (bits == 1))
    
    def test_post_processing_options(self):
        """Test different post-processing options."""
        trajectory = np.random.randn(100, 6)
        
        # No post-processing
        extractor_raw = BitExtractor(post_process=False, debias=False)
        bits_raw = extractor_raw.extract_bits(trajectory)
        
        # With post-processing
        extractor_processed = BitExtractor(post_process=True, debias=True)
        bits_processed = extractor_processed.extract_bits(trajectory)
        
        # Post-processing should generally reduce the number of bits
        assert len(bits_processed) <= len(bits_raw)
    
    def test_von_neumann_debiasing(self):
        """Test von Neumann debiasing specifically."""
        # Create biased sequence (more 1s than 0s)
        biased_bits = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1])
        
        extractor = BitExtractor()
        debiased = extractor._von_neumann_debias(biased_bits)
        
        # Should only keep bits from (0,1) -> 0 and (1,0) -> 1 pairs
        assert len(debiased) <= len(biased_bits) // 2
        assert np.all((debiased == 0) | (debiased == 1))


class TestEntropyAnalyzer:
    """Test the entropy analysis functionality."""
    
    def test_shannon_entropy(self):
        """Test Shannon entropy calculation."""
        analyzer = EntropyAnalyzer()
        
        # Perfect random sequence
        random_bits = np.random.choice([0, 1], size=10000)
        entropy = analyzer.shannon_entropy(random_bits)
        
        # Should be close to 1 for random binary sequence
        assert 0.9 < entropy <= 1.0
        
        # All zeros sequence
        zeros = np.zeros(1000, dtype=int)
        entropy_zeros = analyzer.shannon_entropy(zeros)
        assert entropy_zeros == 0.0
        
        # All ones sequence
        ones = np.ones(1000, dtype=int)
        entropy_ones = analyzer.shannon_entropy(ones)
        assert entropy_ones == 0.0
    
    def test_autocorrelation(self):
        """Test autocorrelation calculation."""
        analyzer = EntropyAnalyzer()
        
        # Random sequence should have low autocorrelation
        random_bits = np.random.choice([0, 1], size=1000)
        autocorr_result = analyzer.autocorrelation(random_bits)
        
        assert 'autocorrelation' in autocorr_result
        assert 'max_autocorr' in autocorr_result
        assert 'independence' in autocorr_result
        
        autocorr = autocorr_result['autocorrelation']
        assert autocorr[0] == 1.0  # Self-correlation should be 1
        assert len(autocorr) <= 101  # Default max_lag is 100
    
    def test_frequency_test(self):
        """Test frequency test."""
        analyzer = EntropyAnalyzer()
        
        # Balanced sequence
        balanced_bits = np.array([0, 1] * 500)
        result = analyzer.frequency_test(balanced_bits)
        
        assert 'uniform' in result
        assert 'p_value' in result
        assert result['uniform'] is True  # Should pass uniformity test
        
        # Unbalanced sequence
        unbalanced_bits = np.array([0] * 900 + [1] * 100)
        result_unbalanced = analyzer.frequency_test(unbalanced_bits)
        
        assert result_unbalanced['uniform'] is False  # Should fail
    
    def test_spectral_test(self):
        """Test spectral analysis."""
        analyzer = EntropyAnalyzer()
        
        # Random sequence
        random_bits = np.random.choice([0, 1], size=1000)
        result = analyzer.spectral_test(random_bits)
        
        assert 'frequencies' in result
        assert 'psd' in result
        assert 'spectral_flatness' in result
        
        # Check that we get reasonable spectral properties
        assert len(result['frequencies']) == len(result['psd'])
        assert result['spectral_flatness'] > 0
    
    def test_entropy_measures(self):
        """Test various entropy measures."""
        analyzer = EntropyAnalyzer()
        
        # Random sequence
        random_bits = np.random.choice([0, 1], size=1000)
        result = analyzer.entropy_measures(random_bits)
        
        assert 'shannon_entropy' in result
        assert 'block_entropy' in result
        assert 'entropy_per_block' in result
        
        # Shannon entropy should be close to 1 for random bits
        assert 0.9 < result['shannon_entropy'] <= 1.0
    
    def test_comprehensive_analysis(self):
        """Test comprehensive analysis function."""
        analyzer = EntropyAnalyzer()
        
        # Generate test sequence
        test_bits = np.random.choice([0, 1], size=5000)
        result = analyzer.comprehensive_analysis(test_bits)
        
        # Check that all analysis components are present
        assert 'sequence_length' in result
        assert 'frequency' in result
        assert 'runs' in result
        assert 'autocorrelation' in result
        assert 'spectral' in result
        assert 'entropy' in result
        assert 'overall_quality' in result
        
        # Check overall quality assessment
        quality = result['overall_quality']
        assert 'tests_passed' in quality
        assert 'total_tests' in quality
        assert 'pass_rate' in quality
        
        assert 0 <= quality['pass_rate'] <= 1.0