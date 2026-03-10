"""
params.py

Contains parameter definitions for the 1D Godunov flash flood simulation.
"""

from dataclasses import dataclass
import numpy as np

# //==================================================\\
# ||               Physical Parameters                ||
# \\==================================================//

@dataclass(frozen=True)
class PhysicalParams:
    """
    Physical parameters for the flux.

    Parameters
    ----------
    g:
        Gravitational acceleration [m/s^2].
    sin_alpha:
        Slope parameter (sin(alpha)) [-]. For small slopes, sin(alpha) ≈ slope.
    f:
        Turbulent friction factor [-].
    """
    g: float = 9.81
    sin_alpha: float = 0.01
    f: float = 0.02
    K_inf: float = 1.38e-6  # ~5 mm/hr for poorly drained desert soil

    def __post_init__(self):
        if self.f <= 0: raise ValueError("f must be positive.")
        if self.sin_alpha < 0: raise ValueError("sin_alpha must be non-negative.")

    @property
    def K(self) -> float:
        """Return K = sqrt(g * sin_alpha / f)."""
        return float(np.sqrt(self.g * self.sin_alpha / self.f))


# //==================================================\\
# ||                Domain Parameters                 ||
# \\==================================================//

@dataclass(frozen=True)
class DomainParams:
    """
    Spatial discretisation parameters.

    Parameters
    ----------
    L:
        Domain length [m], with s in [0, L].
    N:
        Number of finite volumes (cells).
    """
    L: float = 20_000.0
    N: int = 400

    def __post_init__(self):
        if self.N <= 0: raise ValueError("N must be a positive integer.")
        if self.L <= 0: raise ValueError("L must be positive.")

    @property
    def ds(self) -> float:
        """Return grid spacing Δs = L / N [m]."""
        return self.L / self.N

    def cell_centres(self) -> np.ndarray:
        """Return cell-centre coordinates s_i for i=1..N."""
        i = np.arange(1, self.N + 1, dtype=float)
        return (i - 0.5) * self.ds


# //==================================================\\
# ||                 Time Parameters                  ||
# \\==================================================//

@dataclass(frozen=True)
class TimeParams:
    """
    Time discretisation parameters.

    Parameters
    ----------
    dt:
        Constant timestep [s].
    T:
        Final time [s].
    """
    dt: float = 5.0
    T: float = 7_200.0

    def __post_init__(self):
        if self.dt <= 0: raise ValueError("dt must be positive.")
        if self.T < 0: raise ValueError("T must be non-negative.")

    @property
    def n_steps(self) -> int:
        """Return number of timesteps as floor(T / dt)."""
        return int(np.floor(self.T / self.dt))


# //==================================================\\
# ||                Initial Conditions                ||
# \\==================================================//

@dataclass(frozen=True)
class InitialConditionParams:
    """
    Gaussian hump initial condition parameters.

    Parameters
    ----------
    A_base:
        Background wetted area [m^2].
    A_amp:
        Hump amplitude [m^2].
    s0:
        Hump centre location [m].
    sigma:
        Hump width (standard deviation) [m].
    """
    A_base: float = 5.0
    A_amp: float = 20.0
    s0: float = 3_000.0
    sigma: float = 500.0

