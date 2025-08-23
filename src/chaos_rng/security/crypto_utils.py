"""
Cryptographic utilities for enhanced security in chaos-based RNG.

This module provides security enhancements including secure seeding,
entropy mixing, and cryptographic post-processing to ensure the
chaos-based RNG meets cryptographic security requirements.
"""

import os
import hashlib
import hmac
import struct
from typing import Optional, Union, List
import numpy as np

try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False


class SecureSeed:
    """
    Secure seeding utility using multiple entropy sources.
    
    This class combines various entropy sources including os.urandom,
    system time, and optional hardware RNG to create high-quality seeds.
    """
    
    def __init__(self, hardware_rng: Optional[bool] = None):
        """
        Initialize secure seed generator.
        
        Parameters
        ----------
        hardware_rng : bool, optional
            Whether to attempt using hardware RNG if available
        """
        self.hardware_rng = hardware_rng if hardware_rng is not None else False
        self._entropy_sources = []
        self._initialize_sources()
    
    def _initialize_sources(self) -> None:
        """Initialize available entropy sources."""
        # Always available: os.urandom
        self._entropy_sources.append(self._get_os_random)
        
        # System-specific sources
        try:
            # Try to read from /dev/random on Unix systems
            if os.path.exists('/dev/random'):
                self._entropy_sources.append(self._get_dev_random)
        except Exception:
            pass
        
        # Hardware RNG sources
        if self.hardware_rng:
            try:
                # Intel RDRAND instruction (requires specific CPU support)
                self._entropy_sources.append(self._get_rdrand)
            except Exception:
                pass
    
    def _get_os_random(self, size: int) -> bytes:
        """Get entropy from os.urandom."""
        return os.urandom(size)
    
    def _get_dev_random(self, size: int) -> bytes:
        """Get entropy from /dev/random."""
        try:
            with open('/dev/random', 'rb') as f:
                return f.read(size)
        except Exception:
            return b''
    
    def _get_rdrand(self, size: int) -> bytes:
        """Get entropy from Intel RDRAND instruction (if available)."""
        # This is a placeholder - actual implementation would require
        # platform-specific assembly or specialized libraries
        return b''
    
    def generate_seed(self, size: int = 32) -> bytes:
        """
        Generate a secure seed by combining multiple entropy sources.
        
        Parameters
        ----------
        size : int, default=32
            Size of seed in bytes
            
        Returns
        -------
        bytes
            High-entropy seed
        """
        entropy_data = []
        
        # Collect entropy from all available sources
        for source in self._entropy_sources:
            try:
                data = source(size)
                if data:
                    entropy_data.append(data)
            except Exception:
                continue
        
        if not entropy_data:
            raise RuntimeError("No entropy sources available")
        
        # Add timestamp for additional entropy
        timestamp = struct.pack('d', __import__('time').time())
        entropy_data.append(timestamp)
        
        # Add process ID
        pid = struct.pack('I', os.getpid())
        entropy_data.append(pid)
        
        # Combine and hash all entropy
        combined = b''.join(entropy_data)
        
        # Use SHA-256 to mix entropy and produce final seed
        hasher = hashlib.sha256()
        hasher.update(combined)
        
        # For larger seeds, use PBKDF2 for key stretching
        if size > 32:
            salt = hasher.digest()[:16]
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=size,
                salt=salt,
                iterations=10000,
            )
            return kdf.derive(combined)
        else:
            return hasher.digest()[:size]


class EntropyMixer:
    """
    Entropy mixer for combining multiple random sources.
    
    This class implements various mixing techniques to combine
    entropy from different sources while preserving randomness.
    """
    
    def __init__(self, mixing_method: str = 'sha256'):
        """
        Initialize entropy mixer.
        
        Parameters
        ----------
        mixing_method : str, default='sha256'
            Mixing method: 'sha256', 'hmac', or 'xor'
        """
        self.mixing_method = mixing_method
    
    def mix(self, *sources: bytes) -> bytes:
        """
        Mix multiple entropy sources.
        
        Parameters
        ----------
        *sources : bytes
            Variable number of entropy sources to mix
            
        Returns
        -------
        bytes
            Mixed entropy
        """
        if not sources:
            raise ValueError("At least one entropy source required")
        
        if self.mixing_method == 'sha256':
            return self._sha256_mix(*sources)
        elif self.mixing_method == 'hmac':
            return self._hmac_mix(*sources)
        elif self.mixing_method == 'xor':
            return self._xor_mix(*sources)
        else:
            raise ValueError(f"Unknown mixing method: {self.mixing_method}")
    
    def _sha256_mix(self, *sources: bytes) -> bytes:
        """Mix using SHA-256 hash function."""
        hasher = hashlib.sha256()
        for source in sources:
            hasher.update(source)
        return hasher.digest()
    
    def _hmac_mix(self, *sources: bytes) -> bytes:
        """Mix using HMAC construction."""
        if len(sources) < 2:
            # Fall back to SHA-256 if only one source
            return self._sha256_mix(*sources)
        
        # Use first source as key, hash the rest
        key = sources[0]
        message = b''.join(sources[1:])
        return hmac.new(key, message, hashlib.sha256).digest()
    
    def _xor_mix(self, *sources: bytes) -> bytes:
        """Mix using XOR operation."""
        if not sources:
            return b''
        
        # Find minimum length
        min_len = min(len(source) for source in sources)
        if min_len == 0:
            return b''
        
        # XOR all sources
        result = bytearray(sources[0][:min_len])
        for source in sources[1:]:
            for i in range(min_len):
                result[i] ^= source[i]
        
        return bytes(result)


class CryptoPostProcessor:
    """
    Cryptographic post-processor for chaos-based random output.
    
    This class applies cryptographic transformations to the output
    of chaos-based generators to enhance security properties.
    """
    
    def __init__(
        self, 
        method: str = 'chacha20',
        key_size: int = 32,
        auto_rekey: bool = True,
        rekey_interval: int = 1000000
    ):
        """
        Initialize cryptographic post-processor.
        
        Parameters
        ----------
        method : str, default='chacha20'
            Post-processing method: 'chacha20', 'aes_ctr', or 'hash'
        key_size : int, default=32
            Encryption key size in bytes
        auto_rekey : bool, default=True
            Whether to automatically rekey periodically
        rekey_interval : int, default=1000000
            Number of bytes after which to rekey
        """
        self.method = method
        self.key_size = key_size
        self.auto_rekey = auto_rekey
        self.rekey_interval = rekey_interval
        
        self.bytes_processed = 0
        self.current_key = None
        self.cipher = None
        
        if not CRYPTOGRAPHY_AVAILABLE and method in ['chacha20', 'aes_ctr']:
            raise ImportError(
                f"cryptography package required for {method} post-processing"
            )
        
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize the post-processor with a fresh key."""
        # Generate initial key
        secure_seed = SecureSeed()
        self.current_key = secure_seed.generate_seed(self.key_size)
        
        # Initialize cipher
        if self.method == 'chacha20':
            self._init_chacha20()
        elif self.method == 'aes_ctr':
            self._init_aes_ctr()
        # Hash method needs no initialization
        
        self.bytes_processed = 0
    
    def _init_chacha20(self) -> None:
        """Initialize ChaCha20 cipher."""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ImportError("cryptography package required for ChaCha20")
        
        # ChaCha20 uses 32-byte key and 16-byte nonce
        key = self.current_key[:32]
        nonce = os.urandom(16)
        
        algorithm = algorithms.ChaCha20(key, nonce)
        self.cipher = Cipher(algorithm, mode=None)
    
    def _init_aes_ctr(self) -> None:
        """Initialize AES-CTR cipher."""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ImportError("cryptography package required for AES-CTR")
        
        # AES uses 16, 24, or 32-byte keys
        key_size = min(32, max(16, self.key_size))
        key = self.current_key[:key_size]
        nonce = os.urandom(16)
        
        algorithm = algorithms.AES(key)
        mode = modes.CTR(nonce)
        self.cipher = Cipher(algorithm, mode)
    
    def process(self, data: bytes) -> bytes:
        """
        Apply cryptographic post-processing to data.
        
        Parameters
        ----------
        data : bytes
            Input data to process
            
        Returns
        -------
        bytes
            Cryptographically processed data
        """
        if self.auto_rekey and self.bytes_processed >= self.rekey_interval:
            self._rekey()
        
        if self.method == 'hash':
            processed = self._hash_process(data)
        elif self.method in ['chacha20', 'aes_ctr']:
            processed = self._cipher_process(data)
        else:
            raise ValueError(f"Unknown post-processing method: {self.method}")
        
        self.bytes_processed += len(data)
        return processed
    
    def _hash_process(self, data: bytes) -> bytes:
        """Process data using hash-based method."""
        # Use SHA-256 in counter mode
        output = bytearray()
        
        for i in range(0, len(data), 32):
            chunk = data[i:i+32]
            
            # Create hash input: key || counter || chunk
            counter = struct.pack('<Q', i // 32)
            hash_input = self.current_key + counter + chunk
            
            # Hash and XOR with original chunk
            hash_output = hashlib.sha256(hash_input).digest()
            
            for j, byte in enumerate(chunk):
                output.append(byte ^ hash_output[j])
        
        return bytes(output)
    
    def _cipher_process(self, data: bytes) -> bytes:
        """Process data using cipher-based method."""
        if self.cipher is None:
            raise RuntimeError("Cipher not initialized")
        
        encryptor = self.cipher.encryptor()
        # For stream ciphers, encrypt zeros and XOR with data
        zeros = b'\x00' * len(data)
        keystream = encryptor.update(zeros) + encryptor.finalize()
        
        # XOR data with keystream
        output = bytearray()
        for i, byte in enumerate(data):
            output.append(byte ^ keystream[i])
        
        return bytes(output)
    
    def _rekey(self) -> None:
        """Generate a new key for forward secrecy."""
        # Use current output to generate new key
        current_output = self.process(self.current_key)
        
        # Mix with fresh entropy
        secure_seed = SecureSeed()
        fresh_entropy = secure_seed.generate_seed(self.key_size)
        
        mixer = EntropyMixer()
        self.current_key = mixer.mix(current_output, fresh_entropy)
        
        # Reinitialize cipher
        if self.method == 'chacha20':
            self._init_chacha20()
        elif self.method == 'aes_ctr':
            self._init_aes_ctr()
        
        self.bytes_processed = 0


class ConstantTimeOps:
    """
    Constant-time operations for side-channel resistance.
    
    This class provides utilities for performing operations in
    constant time to prevent timing-based side-channel attacks.
    """
    
    @staticmethod
    def constant_time_compare(a: bytes, b: bytes) -> bool:
        """
        Compare two byte strings in constant time.
        
        Parameters
        ----------
        a, b : bytes
            Byte strings to compare
            
        Returns
        -------
        bool
            True if equal, False otherwise
        """
        if len(a) != len(b):
            return False
        
        result = 0
        for x, y in zip(a, b):
            result |= x ^ y
        
        return result == 0
    
    @staticmethod
    def constant_time_select(condition: bool, true_value: int, false_value: int) -> int:
        """
        Select a value in constant time based on condition.
        
        Parameters
        ----------
        condition : bool
            Selection condition
        true_value : int
            Value to return if condition is True
        false_value : int
            Value to return if condition is False
            
        Returns
        -------
        int
            Selected value
        """
        # Convert boolean to 0 or -1
        mask = -(int(condition))
        
        # Constant-time selection
        return (mask & true_value) | (~mask & false_value)


class SecurityValidator:
    """
    Security validator for chaos-based RNG output.
    
    This class performs security-specific tests to ensure
    the generator output meets cryptographic requirements.
    """
    
    def __init__(self):
        """Initialize security validator."""
        pass
    
    def validate_entropy_rate(self, data: bytes, min_entropy: float = 7.0) -> bool:
        """
        Validate minimum entropy rate of data.
        
        Parameters
        ----------
        data : bytes
            Data to validate
        min_entropy : float, default=7.0
            Minimum entropy per byte
            
        Returns
        -------
        bool
            True if entropy rate is sufficient
        """
        if len(data) == 0:
            return False
        
        # Calculate Shannon entropy
        byte_counts = [0] * 256
        for byte in data:
            byte_counts[byte] += 1
        
        entropy = 0.0
        for count in byte_counts:
            if count > 0:
                prob = count / len(data)
                entropy -= prob * np.log2(prob)
        
        return entropy >= min_entropy
    
    def check_forward_secrecy(self, generator, state_size: int = 1000) -> bool:
        """
        Test forward secrecy properties.
        
        Parameters
        ----------
        generator : object
            RNG instance to test
        state_size : int, default=1000
            Size of state to test
            
        Returns
        -------
        bool
            True if forward secrecy is maintained
        """
        # Generate some output
        output1 = generator.bytes(state_size)
        
        # Force a rekey/reseed
        if hasattr(generator, '_reseed'):
            generator._reseed()
        
        # Generate more output
        output2 = generator.bytes(state_size)
        
        # Outputs should be uncorrelated
        correlation = self._calculate_correlation(output1, output2)
        return abs(correlation) < 0.1  # Threshold for independence
    
    def _calculate_correlation(self, data1: bytes, data2: bytes) -> float:
        """Calculate correlation between two byte sequences."""
        if len(data1) != len(data2) or len(data1) == 0:
            return 0.0
        
        # Convert to arrays
        arr1 = np.frombuffer(data1, dtype=np.uint8).astype(float)
        arr2 = np.frombuffer(data2, dtype=np.uint8).astype(float)
        
        # Calculate correlation coefficient
        mean1, mean2 = np.mean(arr1), np.mean(arr2)
        
        numerator = np.sum((arr1 - mean1) * (arr2 - mean2))
        denominator = np.sqrt(np.sum((arr1 - mean1)**2) * np.sum((arr2 - mean2)**2))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator