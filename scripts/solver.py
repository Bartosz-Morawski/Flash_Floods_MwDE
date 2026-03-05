"""
solver.py

A first-order Godunov (finite volume) solver for 1D scalar conservation laws.
"""

from typing import Callable, Dict
import numpy as np
from scripts.params import DomainParams, TimeParams, InitialConditionParams
from scripts.physics import gaussian_initial_condition

# //==================================================\\
# ||                Godunov Simulation                ||
# \\==================================================//

def step_forward_euler_godunov_leftflux(
        A: np.ndarray,
        *,
        dt: float,
        ds: float,
        inflow_A0: float,
        flux: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """
    Advance one timestep using first-order Godunov with left-state flux.

    Parameters
    ----------
    A:
        Cell-average states at current time, shape (N,).
    dt:
        Timestep [s].
    ds:
        Grid spacing [m].
    inflow_A0:
        Upstream inflow boundary state A(0,t) [m^2]. Used as ghost cell A_0.
    flux:
        Function computing Q(A) from A.

    Returns
    -------
    numpy.ndarray
        Updated cell states, shape (N,).
    """
    A = np.asarray(A, dtype=float)

    A_left = np.concatenate(([inflow_A0], A[:-1]))

    Q_i = flux(A)
    Q_im1 = flux(A_left)

    A_new = A - (dt / ds) * (Q_i - Q_im1)
    return np.maximum(A_new, 0.0)


def run_simulation(
        *,
        domain: DomainParams,
        time: TimeParams,
        ic: InitialConditionParams,
        flux_func: Callable[[np.ndarray], np.ndarray],
        store_every: int = 10
) -> Dict[str, np.ndarray]:
    """
    Run a Godunov simulation for a given flux function.

    Parameters
    ----------
    domain:
        Domain parameters (L, N).
    time:
        Time parameters (dt, T).
    ic:
        Gaussian hump parameters.
    flux_func:
        A callable function that computes Q(A).
    store_every:
        Store solution snapshots every this many timesteps.

    Returns
    -------
    dict
        Dictionary containing cell centres ("s"), times ("t"), and history ("A").
    """
    s = domain.cell_centres()
    A = gaussian_initial_condition(s, ic=ic)
    inflow_A0 = ic.A_base

    store_every = max(1, store_every)
    n_store = (time.n_steps // store_every) + 1

    A_hist = np.zeros((n_store, domain.N), dtype=float)
    t_hist = np.zeros((n_store,), dtype=float)

    A_hist[0] = A
    t_hist[0] = 0.0

    k = 1
    t = 0.0

    for n in range(1, time.n_steps + 1):
        A = step_forward_euler_godunov_leftflux(
            A, dt=time.dt, ds=domain.ds, inflow_A0=inflow_A0, flux=flux_func
        )
        t += time.dt

        if n % store_every == 0:
            A_hist[k] = A
            t_hist[k] = t
            k += 1

    return {"s": s, "t": t_hist[:k], "A": A_hist[:k]}