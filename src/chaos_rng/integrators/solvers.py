"""
Numerical integration methods optimized for chaotic systems.

This module provides high-performance ODE solvers specifically designed
for integrating chaotic dynamical systems with good numerical stability.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional

import numba
import numpy as np
from scipy.integrate import solve_ivp


class ODESolver(ABC):
    """Abstract base class for ODE solvers."""

    @abstractmethod
    def integrate(
        self,
        f: Callable[[float, np.ndarray], np.ndarray],
        y0: np.ndarray,
        t_span: tuple[float, float],
        max_step: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Integrate ODE system from t_span[0] to t_span[1]."""
        pass


class RK45Solver(ODESolver):
    """
    Runge-Kutta 4th/5th order adaptive step solver.

    This solver provides excellent performance for chaotic systems
    with automatic step size control for numerical stability.
    """

    def __init__(self, rtol: float = 1e-9, atol: float = 1e-12):
        """
        Initialize RK45 solver with tolerance parameters.

        Parameters
        ----------
        rtol : float, default=1e-9
            Relative tolerance for integration
        atol : float, default=1e-12
            Absolute tolerance for integration
        """
        self.rtol = rtol
        self.atol = atol

    def integrate(
        self,
        f: Callable[[float, np.ndarray], np.ndarray],
        y0: np.ndarray,
        t_span: tuple[float, float],
        max_step: Optional[float] = None,
        dense_output: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Integrate ODE system using RK45 method.

        Parameters
        ----------
        f : callable
            Right-hand side of the ODE system: dy/dt = f(t, y)
        y0 : array_like
            Initial state vector
        t_span : tuple
            Integration time span (t_start, t_end)
        max_step : float, optional
            Maximum allowed step size
        dense_output : bool, default=False
            Whether to provide dense output for interpolation

        Returns
        -------
        t : ndarray
            Time points
        y : ndarray
            Solution values at time points
        """
        # Handle max_step parameter for SciPy compatibility
        solver_kwargs = {
            "method": "RK45",
            "rtol": self.rtol,
            "atol": self.atol,
            "dense_output": dense_output,
        }
        if max_step is not None:
            solver_kwargs["max_step"] = max_step

        result = solve_ivp(f, t_span, y0, **solver_kwargs)

        if not result.success:
            raise RuntimeError(f"Integration failed: {result.message}")

        return result.t, result.y.T


class AdaptiveSolver(ODESolver):
    """
    Custom adaptive solver optimized for chaotic systems.

    This solver uses Numba JIT compilation for maximum performance
    and includes specialized error control for chaotic dynamics.
    """

    def __init__(self, rtol: float = 1e-9, atol: float = 1e-12):
        """
        Initialize adaptive solver.

        Parameters
        ----------
        rtol : float, default=1e-9
            Relative tolerance
        atol : float, default=1e-12
            Absolute tolerance
        """
        self.rtol = rtol
        self.atol = atol

    def integrate(
        self,
        f: Callable[[float, np.ndarray], np.ndarray],
        y0: np.ndarray,
        t_span: tuple[float, float],
        max_step: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Integrate using custom adaptive method.

        Parameters
        ----------
        f : callable
            Right-hand side function
        y0 : array_like
            Initial conditions
        t_span : tuple
            Time span
        max_step : float, optional
            Maximum step size

        Returns
        -------
        t : ndarray
            Time points
        y : ndarray
            Solution trajectory
        """
        # Use optimized Numba implementation
        return _adaptive_rk45_jit(
            f,
            y0.astype(np.float64),
            t_span[0],
            t_span[1],
            self.rtol,
            self.atol,
            max_step,
        )


@numba.njit(cache=True)
def _rk45_step(
    f_func, t: float, y: np.ndarray, h: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Single RK45 step with error estimation.

    This is a JIT-compiled implementation of the Runge-Kutta-Fehlberg
    method for maximum performance in chaotic system integration.
    """
    # RK45 coefficients
    a2, a3, a4, a5, a6 = 1 / 4, 3 / 8, 12 / 13, 1.0, 1 / 2

    b21 = 1 / 4
    b31, b32 = 3 / 32, 9 / 32
    b41, b42, b43 = 1932 / 2197, -7200 / 2197, 7296 / 2197
    b51, b52, b53, b54 = 439 / 216, -8, 3680 / 513, -845 / 4104
    b61, b62, b63, b64, b65 = -8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40

    # 4th order coefficients
    c1, c3, c4, c5 = 25 / 216, 1408 / 2565, 2197 / 4104, -1 / 5

    # 5th order coefficients
    d1, d3, d4, d5, d6 = 16 / 135, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55

    # Compute k values
    k1 = h * f_func(t, y)
    k2 = h * f_func(t + a2 * h, y + b21 * k1)
    k3 = h * f_func(t + a3 * h, y + b31 * k1 + b32 * k2)
    k4 = h * f_func(t + a4 * h, y + b41 * k1 + b42 * k2 + b43 * k3)
    k5 = h * f_func(t + a5 * h, y + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4)
    k6 = h * f_func(
        t + a6 * h, y + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5
    )

    # 4th and 5th order solutions
    y4 = y + c1 * k1 + c3 * k3 + c4 * k4 + c5 * k5
    y5 = y + d1 * k1 + d3 * k3 + d4 * k4 + d5 * k5 + d6 * k6

    # Error estimate
    error = np.linalg.norm(y5 - y4)

    return y5, y4, error


@numba.njit(cache=True)
def _adaptive_rk45_jit(
    f_func,
    y0: np.ndarray,
    t0: float,
    tf: float,
    rtol: float,
    atol: float,
    max_step: Optional[float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    JIT-compiled adaptive RK45 integration.

    This provides maximum performance for repeated integration
    of chaotic systems.
    """
    if max_step is None:
        max_step = (tf - t0) / 1000

    # Initialize arrays
    t_list = [t0]
    y_list = [y0.copy()]

    t = t0
    y = y0.copy()
    h = min(max_step, (tf - t0) / 100)  # Initial step size

    safety = 0.84
    min_factor = 0.1
    max_factor = 4.0

    while t < tf:
        # Don't overshoot final time
        if t + h > tf:
            h = tf - t

        # Take RK45 step
        y_new, y_old, error = _rk45_step(f_func, t, y, h)

        # Error control
        tol = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        error_norm = error / np.linalg.norm(tol)

        if error_norm <= 1.0:
            # Accept step
            t += h
            y = y_new
            t_list.append(t)
            y_list.append(y.copy())

            # Update step size
            if error_norm > 0:
                factor = safety * (1.0 / error_norm) ** 0.2
                factor = max(min_factor, min(factor, max_factor))
                h *= factor
            h = min(h, max_step)
        else:
            # Reject step and reduce step size
            factor = safety * (1.0 / error_norm) ** 0.25
            h *= max(min_factor, factor)

    # Convert to arrays
    t_array = np.array(t_list)
    y_array = np.array(y_list)

    return t_array, y_array


class LyapunovCalculator:
    """
    Calculator for Lyapunov exponents of chaotic systems.

    Lyapunov exponents quantify the rate of separation of infinitesimally
    close trajectories, providing a measure of chaotic behavior.
    """

    def __init__(self, solver: ODESolver):
        """
        Initialize Lyapunov calculator.

        Parameters
        ----------
        solver : ODESolver
            ODE solver to use for integration
        """
        self.solver = solver

    def compute_largest_exponent(
        self,
        f: Callable[[float, np.ndarray], np.ndarray],
        jac: Callable[[float, np.ndarray], np.ndarray],
        y0: np.ndarray,
        t_max: float,
        dt: float = 0.01,
    ) -> float:
        """
        Compute the largest Lyapunov exponent.

        Parameters
        ----------
        f : callable
            System dynamics function
        jac : callable
            Jacobian matrix function
        y0 : array_like
            Initial conditions
        t_max : float
            Integration time
        dt : float, default=0.01
            Renormalization interval

        Returns
        -------
        float
            Largest Lyapunov exponent
        """
        n_dim = len(y0)
        n_steps = int(t_max / dt)

        # Initialize with random perturbation
        delta = np.random.randn(n_dim)
        delta = delta / np.linalg.norm(delta) * 1e-8

        lyap_sum = 0.0
        y = y0.copy()

        for i in range(n_steps):
            # Integrate main trajectory
            t_span = (i * dt, (i + 1) * dt)
            _, y_traj = self.solver.integrate(f, y, t_span)
            y = y_traj[-1]

            # Evolve perturbation using linearization
            J = jac(i * dt, y)
            delta = J @ delta * dt + delta

            # Renormalize and accumulate
            norm = np.linalg.norm(delta)
            if norm > 0:
                lyap_sum += np.log(norm / 1e-8)
                delta = delta / norm * 1e-8

        return lyap_sum / t_max
