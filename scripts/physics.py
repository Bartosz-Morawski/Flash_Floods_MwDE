"""
physics.py

Mathematical models for river bed geometries, flux closures, and initial conditions.
"""

from typing import Callable
import numpy as np
from scripts.params import PhysicalParams, InitialConditionParams

# //==================================================\\
# ||                    Box Canyon                    ||
# \\==================================================//

def l_rectangular(A: np.ndarray, *, w: float) -> np.ndarray:
    """
    Wetted perimeter l(A) for a rectangular channel.

    Parameters
    ----------
    A:
        Wetted cross-sectional area(s) [m^2]. Must be >= 0.
    w:
        Channel width [m], w > 0.

    Returns
    -------
    numpy.ndarray
        Wetted perimeter(s) l(A) [m].
    """
    if w <= 0: raise ValueError("Width w must be positive.")
    A = np.asarray(A, dtype=float)
    return w + (2.0 * A) / w


# //==================================================\\
# ||                       Flux                       ||
# \\==================================================//

def flux_Q(
    A: np.ndarray,
    *,
    phys: PhysicalParams,
    l_of_A: Callable[[np.ndarray], np.ndarray],
    eps: float = 1e-12
) -> np.ndarray:
    """
    Compute the flux Q(A) = K * A^(3/2) / sqrt(l(A)).

    Parameters
    ----------
    A:
        Wetted cross-sectional area(s) [m^2].
    phys:
        Physical parameters (defines K).
    l_of_A:
        Function returning wetted perimeter l(A) [m] for given A.
    eps:
        Small number to prevent division by zero.

    Returns
    -------
    numpy.ndarray
        Flux values Q(A) [m^3/s].
    """
    A = np.asarray(A, dtype=float)
    A_pos = np.maximum(A, 0.0)
    lA = np.maximum(np.asarray(l_of_A(A_pos), dtype=float), eps)
    return phys.K * (A_pos ** 1.5) / np.sqrt(lA)


# //==================================================\\
# ||                   Normal Curve                   ||
# \\==================================================//

def gaussian_initial_condition(
    s: np.ndarray,
    *,
    ic: InitialConditionParams
) -> np.ndarray:
    """
    Construct Gaussian hump initial condition A(s,0).

    Parameters
    ----------
    s:
        Spatial locations [m].
    ic:
        Initial condition parameters.

    Returns
    -------
    numpy.ndarray
        Initial A values [m^2] at the provided s locations.
    """
    if ic.sigma <= 0: raise ValueError("sigma must be positive.")
    return ic.A_base + ic.A_amp * np.exp(-((s - ic.s0) ** 2) / (2.0 * ic.sigma ** 2))

def wave_speed_rectangular(A: np.ndarray, *, phys: PhysicalParams, w: float) -> np.ndarray:
    """
    Calculate the characteristic wave speed v(A) = Q'(A) for a rectangular canyon.
    For Q(A) = K * A^(3/2) / sqrt(w), the derivative is (3/2) * K * A^(1/2) / sqrt(w).
    """
    A_pos = np.maximum(np.asarray(A, dtype=float), 0.0)
    return (3.0 / 2.0) * phys.K * (A_pos ** 0.5) / np.sqrt(w)