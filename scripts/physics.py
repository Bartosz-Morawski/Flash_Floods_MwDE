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
    Calculate the exact characteristic wave speed v(A) = Q'(A) for a rectangular canyon.
    Using the full wetted perimeter l(A) = w + 2A/w.
    """
    A_pos = np.maximum(np.asarray(A, dtype=float), 0.0)

    # Calculate exact wetted perimeter
    l_A = w + (2.0 * A_pos) / w

    # Calculate exact derivative Q'(A)
    term1 = 1.5 * np.sqrt(A_pos) / np.sqrt(l_A)
    term2 = (A_pos ** 1.5) / (w * (l_A ** 1.5))

    return phys.K * (term1 - term2)


# def calculate_breaking_time(
#         A0: np.ndarray,
#         s: np.ndarray,
#         *,
#         phys: PhysicalParams,
#         w: float
# ) -> float:
#     """
#     Calculate the exact time t* of gradient catastrophe (wave breaking)
#     where the characteristics first intersect.
#
#     Parameters
#     ----------
#     A0:
#         Initial wetted cross-sectional area(s) [m^2].
#     s:
#         Spatial locations [m].
#     phys:
#         Physical parameters.
#     w:
#         Channel width [m].
#
#     Returns
#     -------
#     float
#         Breaking time t* [s]. Returns np.inf if the wave never breaks.
#     """
#     # 1. Calculate initial wave speed using our existing exact function
#     v0 = wave_speed_rectangular(A0, phys=phys, w=w)
#
#     # 2. Calculate the spatial gradient of the velocity
#     dv0_ds = np.gradient(v0, s)
#     min_grad = np.min(dv0_ds)
#
#     # 3. Calculate breaking time
#     if min_grad < 0.0:
#         return float(-1.0 / min_grad)
#     else:
#         return float(np.inf)

def calculate_breaking_time(
        A0: np.ndarray,
        s: np.ndarray,
        *,
        phys: PhysicalParams,
        w: float
) -> float:
    """
    Calculate the exact time t* of gradient catastrophe by finding
    the absolute first point where two adjacent characteristic lines intersect.
    """
    # 1. Calculate initial wave speeds
    v0 = wave_speed_rectangular(A0, phys=phys, w=w)

    # 2. Calculate distance (P2 - P1) and speed differences (v1 - v2)
    # between all adjacent points on the grid
    ds = np.diff(s)
    dv = v0[:-1] - v0[1:]

    # 3. We only care about points that are crashing into each other (v1 > v2)
    converging = dv > 0

    # 4. Calculate all intersection times and find the earliest one
    if np.any(converging):
        t_crossings = ds[converging] / dv[converging]
        return float(np.min(t_crossings))
    else:
        return float(np.inf)  # Wave never breaks