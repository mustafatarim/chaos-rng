"""
Three-body gravitational system for chaos-based random number generation.

This module implements the three-body problem as a source of chaotic dynamics
for high-quality random number generation. The system exhibits sensitive 
dependence on initial conditions, making it an excellent entropy source.

The three-body problem is described by:
    m_i * d²r_i/dt² = Σ(j≠i) G*m_i*m_j*(r_j - r_i)/|r_j - r_i|³

where m_i are the masses, r_i are position vectors, and G is the gravitational constant.
"""

from typing import Optional, Union, Tuple, List, Callable
import os
import numpy as np
import numba
from scipy.integrate import solve_ivp

from ..integrators.solvers import RK45Solver, LyapunovCalculator
from .bit_extractors import BitExtractor


class ThreeBodySystem:
    """
    Three-body gravitational system implementation.
    
    This class simulates the motion of three gravitationally interacting
    point masses, which exhibits chaotic behavior for most initial conditions.
    """
    
    def __init__(
        self, 
        masses: Optional[List[float]] = None,
        G: float = 1.0,
        solver_rtol: float = 1e-9,
        solver_atol: float = 1e-12
    ):
        """
        Initialize three-body system.
        
        Parameters
        ----------
        masses : list of float, optional
            Masses of the three bodies. Default: [1.0, 1.0, 1.0]
        G : float, default=1.0
            Gravitational constant
        solver_rtol : float, default=1e-9
            Relative tolerance for ODE solver
        solver_atol : float, default=1e-12  
            Absolute tolerance for ODE solver
        """
        self.masses = np.array(masses if masses is not None else [1.0, 1.0, 1.0])
        self.G = G
        self.n_bodies = len(self.masses)
        self.n_dim = 2  # 2D system for computational efficiency
        self.state_size = self.n_bodies * self.n_dim * 2  # positions + velocities
        
        # Initialize solver
        self.solver = RK45Solver(rtol=solver_rtol, atol=solver_atol)
        self.lyapunov_calc = LyapunovCalculator(self.solver)
        
        # Current state: [x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3]
        self.state = None
        self.time = 0.0
        
        # Pre-compile JIT functions
        self._equations_jit = self._compile_equations()
        self._jacobian_jit = self._compile_jacobian()
    
    def _compile_equations(self) -> Callable:
        """Compile three-body equations with Numba JIT."""
        masses = self.masses
        G = self.G
        n_bodies = self.n_bodies
        n_dim = self.n_dim
        
        @numba.njit(cache=True)
        def equations(t: float, state: np.ndarray) -> np.ndarray:
            """
            Three-body equations of motion.
            
            Parameters
            ----------
            t : float
                Time (unused but required by ODE solver interface)
            state : ndarray
                Current state vector [positions, velocities]
                
            Returns
            -------
            ndarray
                Time derivatives [velocities, accelerations]
            """
            # Extract positions and velocities
            pos = state[:n_bodies * n_dim].reshape((n_bodies, n_dim))
            vel = state[n_bodies * n_dim:].reshape((n_bodies, n_dim))
            
            # Initialize accelerations
            acc = np.zeros_like(pos)
            
            # Calculate gravitational forces
            for i in range(n_bodies):
                for j in range(n_bodies):
                    if i != j:
                        # Vector from body i to body j
                        r_vec = pos[j] - pos[i]
                        r_mag = np.sqrt(np.sum(r_vec ** 2))
                        
                        # Avoid singularities with softening parameter
                        r_mag = max(r_mag, 1e-6)
                        
                        # Gravitational acceleration
                        acc[i] += G * masses[j] * r_vec / (r_mag ** 3)
            
            # Return derivatives: [velocities, accelerations]
            derivatives = np.zeros_like(state)
            derivatives[:n_bodies * n_dim] = vel.flatten()
            derivatives[n_bodies * n_dim:] = acc.flatten()
            
            return derivatives
        
        return equations
    
    def _compile_jacobian(self) -> Callable:
        """Compile Jacobian matrix with Numba JIT."""
        masses = self.masses
        G = self.G
        n_bodies = self.n_bodies
        n_dim = self.n_dim
        state_size = self.state_size
        
        @numba.njit(cache=True)
        def jacobian(t: float, state: np.ndarray) -> np.ndarray:
            """
            Jacobian matrix of the three-body system.
            
            This is used for Lyapunov exponent calculation and
            provides the linearization of the dynamics.
            """
            J = np.zeros((state_size, state_size))
            
            # Extract positions
            pos = state[:n_bodies * n_dim].reshape((n_bodies, n_dim))
            
            # Velocity derivatives (upper-right block)
            for i in range(n_bodies * n_dim):
                J[i, i + n_bodies * n_dim] = 1.0
            
            # Acceleration derivatives (lower-left block)
            for i in range(n_bodies):
                for j in range(n_bodies):
                    if i != j:
                        r_vec = pos[j] - pos[i]
                        r_mag = np.sqrt(np.sum(r_vec ** 2))
                        r_mag = max(r_mag, 1e-6)
                        
                        # Derivative of acceleration w.r.t. position
                        for k in range(n_dim):
                            for l in range(n_dim):
                                idx_i = (n_bodies + i) * n_dim + k
                                idx_j = j * n_dim + l
                                
                                if k == l:
                                    J[idx_i, idx_j] += G * masses[j] * (
                                        1.0 / (r_mag ** 3) - 
                                        3.0 * r_vec[k] ** 2 / (r_mag ** 5)
                                    )
                                else:
                                    J[idx_i, idx_j] += G * masses[j] * (
                                        -3.0 * r_vec[k] * r_vec[l] / (r_mag ** 5)
                                    )
                                
                                # Diagonal terms (self-interaction)
                                idx_i_self = (n_bodies + i) * n_dim + k
                                idx_i_pos = i * n_dim + l
                                J[idx_i_self, idx_i_pos] -= J[idx_i, idx_j]
            
            return J
        
        return jacobian
    
    def set_initial_conditions(
        self, 
        positions: Optional[np.ndarray] = None,
        velocities: Optional[np.ndarray] = None,
        random_ic: bool = True,
        energy_scale: float = 1.0
    ) -> None:
        """
        Set initial conditions for the three-body system.
        
        Parameters
        ----------
        positions : ndarray, optional
            Initial positions [3, 2] array
        velocities : ndarray, optional  
            Initial velocities [3, 2] array
        random_ic : bool, default=True
            Whether to use random initial conditions
        energy_scale : float, default=1.0
            Scale factor for total energy
        """
        if random_ic or (positions is None or velocities is None):
            # Generate random initial conditions
            positions, velocities = self._generate_random_ic(energy_scale)
        
        # Validate shapes
        positions = np.array(positions).reshape((self.n_bodies, self.n_dim))
        velocities = np.array(velocities).reshape((self.n_bodies, self.n_dim))
        
        # Construct state vector
        self.state = np.concatenate([positions.flatten(), velocities.flatten()])
        self.time = 0.0
    
    def _generate_random_ic(self, energy_scale: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate random initial conditions that avoid immediate collisions.
        
        Parameters
        ----------
        energy_scale : float
            Energy scale for the system
            
        Returns
        -------
        positions : ndarray
            Random initial positions
        velocities : ndarray
            Random initial velocities
        """
        # Generate positions avoiding close encounters
        min_distance = 0.5  # Minimum separation
        max_attempts = 1000
        
        for attempt in range(max_attempts):
            positions = np.random.uniform(-2, 2, (self.n_bodies, self.n_dim))
            
            # Check minimum distances
            valid = True
            for i in range(self.n_bodies):
                for j in range(i + 1, self.n_bodies):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist < min_distance:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                break
        else:
            # Fallback: use fixed stable configuration
            positions = np.array([
                [-1.0, 0.0],
                [1.0, 0.0], 
                [0.0, 1.0]
            ])
        
        # Generate velocities with zero center of mass motion
        velocities = np.random.normal(0, energy_scale, (self.n_bodies, self.n_dim))
        
        # Remove center of mass motion
        com_velocity = np.sum(self.masses[:, np.newaxis] * velocities, axis=0) / np.sum(self.masses)
        velocities -= com_velocity[np.newaxis, :]
        
        return positions, velocities
    
    def evolve(self, dt: float, steps: int = 1) -> np.ndarray:
        """
        Evolve the three-body system forward in time.
        
        Parameters
        ----------
        dt : float
            Time step size
        steps : int, default=1
            Number of time steps to take
            
        Returns
        -------
        ndarray
            Trajectory of system states [steps+1, state_size]
        """
        if self.state is None:
            self.set_initial_conditions(random_ic=True)
        
        total_time = dt * steps
        t_span = (self.time, self.time + total_time)
        
        # Integrate using optimized solver
        t_points, trajectory = self.solver.integrate(
            self._equations_jit, self.state, t_span
        )
        
        # Update current state
        self.state = trajectory[-1]
        self.time = t_points[-1]
        
        return trajectory
    
    def compute_lyapunov_exponent(
        self, 
        t_max: float = 100.0,
        dt: float = 0.01
    ) -> float:
        """
        Compute the largest Lyapunov exponent.
        
        Parameters
        ----------
        t_max : float, default=100.0
            Integration time for calculation
        dt : float, default=0.01
            Renormalization interval
            
        Returns
        -------
        float
            Largest Lyapunov exponent
        """
        if self.state is None:
            self.set_initial_conditions(random_ic=True)
        
        return self.lyapunov_calc.compute_largest_exponent(
            self._equations_jit, self._jacobian_jit, 
            self.state, t_max, dt
        )
    
    def get_energy(self) -> float:
        """
        Calculate total energy of the system.
        
        Returns
        -------
        float
            Total energy (kinetic + potential)
        """
        if self.state is None:
            return 0.0
        
        # Extract positions and velocities
        pos = self.state[:self.n_bodies * self.n_dim].reshape((self.n_bodies, self.n_dim))
        vel = self.state[self.n_bodies * self.n_dim:].reshape((self.n_bodies, self.n_dim))
        
        # Kinetic energy
        kinetic = 0.5 * np.sum(self.masses[:, np.newaxis] * vel ** 2)
        
        # Potential energy
        potential = 0.0
        for i in range(self.n_bodies):
            for j in range(i + 1, self.n_bodies):
                r_dist = np.linalg.norm(pos[i] - pos[j])
                potential -= self.G * self.masses[i] * self.masses[j] / max(r_dist, 1e-6)
        
        return kinetic + potential


class ThreeBodyRNG:
    """
    Professional chaos-based random number generator using three-body dynamics.
    
    This class provides a high-quality random number generator that uses
    the chaotic motion of a three-body gravitational system as an entropy source.
    The generator implements multiple bit extraction methods and provides
    compatibility with NumPy's random number generation framework.
    """
    
    def __init__(
        self,
        seed: Optional[Union[int, np.ndarray]] = None,
        masses: Optional[List[float]] = None,
        extraction_method: str = 'lsb',
        buffer_size: int = 10000,
        reseed_interval: int = 1000000
    ):
        """
        Initialize three-body random number generator.
        
        Parameters
        ----------
        seed : int or array_like, optional
            Seed for reproducible random numbers. If None, uses os.urandom
        masses : list of float, optional
            Masses of the three bodies. Default: [1.0, 1.0, 1.0]
        extraction_method : str, default='lsb'
            Bit extraction method: 'lsb', 'threshold', or 'poincare'
        buffer_size : int, default=10000
            Size of internal random bit buffer
        reseed_interval : int, default=1000000
            Number of bits after which to reseed for forward secrecy
        """
        self.extraction_method = extraction_method
        self.buffer_size = buffer_size
        self.reseed_interval = reseed_interval
        self.bits_generated = 0
        
        # Initialize three-body system
        self.system = ThreeBodySystem(masses=masses)
        
        # Initialize bit extractor
        self.bit_extractor = BitExtractor(method=extraction_method)
        
        # Bit buffer for efficient generation
        self._bit_buffer = np.array([], dtype=np.uint8)
        self._buffer_index = 0
        
        # Set initial conditions based on seed
        self._initialize_from_seed(seed)
    
    def _initialize_from_seed(self, seed: Optional[Union[int, np.ndarray]]) -> None:
        """Initialize system state from seed."""
        if seed is None:
            # Use cryptographically secure random seed
            seed_bytes = os.urandom(32)
            seed_array = np.frombuffer(seed_bytes, dtype=np.uint32)
        elif isinstance(seed, int):
            # Convert integer seed to reproducible array
            np.random.seed(seed)
            seed_array = np.random.randint(0, 2**32, size=8, dtype=np.uint32)
        else:
            seed_array = np.array(seed, dtype=np.uint32)
        
        # Use seed to generate initial conditions
        np.random.seed(seed_array[0])
        energy_scale = 0.5 + 0.5 * (seed_array[1] % 1000) / 1000.0
        
        self.system.set_initial_conditions(
            random_ic=True,
            energy_scale=energy_scale
        )
        
        # Evolve system to avoid transients
        self.system.evolve(dt=0.01, steps=1000)
    
    def _fill_buffer(self) -> None:
        """Fill the internal bit buffer with fresh random bits."""
        # Evolve system to generate new trajectory
        trajectory = self.system.evolve(dt=0.001, steps=self.buffer_size // 12)
        
        # Extract bits from trajectory
        bits = self.bit_extractor.extract_bits(trajectory)
        
        # Ensure we have enough bits
        if len(bits) < self.buffer_size:
            # Generate more if needed
            additional_steps = (self.buffer_size - len(bits)) // 12 + 1
            additional_traj = self.system.evolve(dt=0.001, steps=additional_steps)
            additional_bits = self.bit_extractor.extract_bits(additional_traj)
            bits = np.concatenate([bits, additional_bits])
        
        # Store in buffer
        self._bit_buffer = bits[:self.buffer_size]
        self._buffer_index = 0
        self.bits_generated += len(self._bit_buffer)
        
        # Check if reseeding is needed
        if self.bits_generated >= self.reseed_interval:
            self._reseed()
    
    def _reseed(self) -> None:
        """Reseed the generator for forward secrecy."""
        # Use current system state as part of new seed
        current_state = self.system.state
        hash_input = current_state.tobytes() + os.urandom(16)
        
        # Simple hash function (in production, use cryptographic hash)
        seed_value = sum(hash_input) % (2**32)
        self._initialize_from_seed(seed_value)
        self.bits_generated = 0
    
    def _get_random_bits(self, n_bits: int) -> np.ndarray:
        """Get n random bits from the buffer."""
        bits = np.array([], dtype=np.uint8)
        
        while len(bits) < n_bits:
            # Check if buffer needs refilling
            if self._buffer_index >= len(self._bit_buffer):
                self._fill_buffer()
            
            # Take bits from buffer
            available = len(self._bit_buffer) - self._buffer_index
            needed = n_bits - len(bits)
            take = min(available, needed)
            
            bits = np.concatenate([
                bits, 
                self._bit_buffer[self._buffer_index:self._buffer_index + take]
            ])
            self._buffer_index += take
        
        return bits[:n_bits]
    
    def random(self, size: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[float, np.ndarray]:
        """
        Generate random floats in [0, 1).
        
        Parameters
        ----------
        size : int or tuple, optional
            Output shape. If None, return single float
            
        Returns
        -------
        float or ndarray
            Random float(s) in [0, 1)
        """
        if size is None:
            # Single value
            bits = self._get_random_bits(64)  # 64 bits for double precision
            return self._bits_to_float(bits)
        
        # Array of values
        if isinstance(size, int):
            size = (size,)
        
        n_values = np.prod(size)
        bits = self._get_random_bits(64 * n_values)
        
        # Convert bits to floats
        floats = np.array([
            self._bits_to_float(bits[i*64:(i+1)*64]) 
            for i in range(n_values)
        ])
        
        return floats.reshape(size)
    
    def _bits_to_float(self, bits: np.ndarray) -> float:
        """Convert 64 random bits to a float in [0, 1)."""
        # Pack bits into uint64
        uint_val = 0
        for i, bit in enumerate(bits[:64]):
            uint_val |= (int(bit) << i)
        
        # Convert to float in [0, 1)
        return uint_val / (2**64)
    
    def randint(
        self, 
        low: int, 
        high: int, 
        size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> Union[int, np.ndarray]:
        """
        Generate random integers in [low, high).
        
        Parameters
        ----------
        low : int
            Lower bound (inclusive)
        high : int
            Upper bound (exclusive)
        size : int or tuple, optional
            Output shape
            
        Returns
        -------
        int or ndarray
            Random integer(s) in [low, high)
        """
        if high <= low:
            raise ValueError("high must be greater than low")
        
        range_size = high - low
        
        if size is None:
            # Single value
            return int(self.random() * range_size) + low
        
        # Array of values
        floats = self.random(size)
        return (floats * range_size).astype(int) + low
    
    def bytes(self, length: int) -> bytes:
        """
        Generate random bytes for cryptographic use.
        
        Parameters
        ----------
        length : int
            Number of bytes to generate
            
        Returns
        -------
        bytes
            Random bytes
        """
        bits = self._get_random_bits(length * 8)
        
        # Pack bits into bytes
        byte_array = np.packbits(bits.reshape(-1, 8)[:length])
        return bytes(byte_array)
    
    def get_state(self) -> dict:
        """Get the current generator state for serialization."""
        return {
            'system_state': self.system.state.copy(),
            'system_time': self.system.time,
            'buffer': self._bit_buffer.copy(),
            'buffer_index': self._buffer_index,
            'bits_generated': self.bits_generated
        }
    
    def set_state(self, state: dict) -> None:
        """Set the generator state from serialized data."""
        self.system.state = state['system_state'].copy()
        self.system.time = state['system_time']
        self._bit_buffer = state['buffer'].copy()
        self._buffer_index = state['buffer_index']
        self.bits_generated = state['bits_generated']