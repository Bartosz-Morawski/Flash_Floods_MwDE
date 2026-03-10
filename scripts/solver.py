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

import numpy as np
from typing import Dict, Callable


def run_simulation_infiltration(
        domain: 'DomainParams',
        time: 'TimeParams',
        ic: 'InitialConditionParams',
        phys: 'PhysicalParams',
        w_rect: float,
        flux_func: Callable,
        store_every: int = 10
) -> Dict[str, np.ndarray]:
    """
    Solves the Kinematic Wave Equation WITH a soil infiltration sink term
    using Operator Splitting (Godunov Advection + Explicit Sink).
    """
    # 1. Setup Grid and Arrays (Same as your original solver)
    s = np.linspace(0.0, domain.L, domain.N)
    ds = s[1] - s[0]
    dt = time.dt
    Nt = int(time.T / dt)

    # Initialize A array
    from scripts.physics import gaussian_initial_condition
    A = gaussian_initial_condition(s, ic=ic)
    # Storage arrays
    store_count = Nt // store_every + 1
    A_hist = np.zeros((store_count, domain.N))
    t_hist = np.zeros(store_count)

    A_hist[0, :] = A.copy()
    t_hist[0] = 0.0
    store_idx = 1

    # 2. The Fractional Step Time Loop
    for n in range(1, Nt + 1):

        # --- STEP A: Godunov Advection (Move the water) ---
        # Calculate fluxes at cell interfaces (Upwind scheme)
        F = np.zeros(domain.N + 1)
        for i in range(1, domain.N):
            F[i] = flux_func(A[i - 1])

            # Boundary conditions
        F[0] = flux_func(ic.A_base)  # Trickle coming in
        F[-1] = F[-2]  # Outflow boundary

        # Intermediate update (A_star)
        A_star = np.zeros_like(A)
        for i in range(domain.N):
            A_star[i] = A[i] - (dt / ds) * (F[i + 1] - F[i])

        # --- STEP B: Soil Infiltration (Drain the water) ---
        # Calculate wetted perimeter: l(A) = w + 2A/w
        l_A = w_rect + (2.0 * A_star) / w_rect

        # Calculate the sink volume: I(A) = K_inf * l(A)
        sink_term = phys.K_inf * l_A

        # Update Area
        A_new = A_star - (dt * sink_term)

        # CRITICAL FIX: Prevent the "Dry Bed Singularity"
        # We cannot let the soil drain the river below our baseline trickle
        A_new = np.maximum(A_new, ic.A_base)

        # Overwrite A for the next loop
        A = A_new.copy()

        # Store results
        if n % store_every == 0 or n == Nt:
            A_hist[store_idx, :] = A.copy()
            t_hist[store_idx] = n * dt
            store_idx += 1

    return {"t": t_hist[:store_idx], "s": s, "A": A_hist[:store_idx, :]}