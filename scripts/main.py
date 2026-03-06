"""
main.py

Entry point for the flash-flood Godunov simulation.
Generates diagnostics and plots.
"""

import numpy as np
from params import DomainParams, TimeParams, PhysicalParams, InitialConditionParams
from physics import l_rectangular, flux_Q, gaussian_initial_condition, wave_speed_rectangular
from solver import run_simulation
import plotting

def main() -> None:
    # 1. Setup Configuration
    domain = DomainParams(L=50_000.0, N=1000)
    time = TimeParams(dt=5.0, T=14_400.0) # 4 Hours
    phys = PhysicalParams(g=9.81, sin_alpha=0.01, f=0.02)
    ic = InitialConditionParams(A_base=5.0, A_amp=20.0, s0=3_000.0, sigma=500.0)
    w_rect = 20.0 # Box canyon width

    # 2. Define the baseline flux function (Rectangular Box Canyon)
    def flux_rect(A_in: np.ndarray) -> np.ndarray:
        return flux_Q(A_in, phys=phys, l_of_A=lambda a: l_rectangular(a, w=w_rect))

    # 3. Run the baseline simulation
    print("Running Godunov Simulation...")
    out = run_simulation(
        domain=domain, time=time, ic=ic, flux_func=flux_rect, store_every=20
    )

    print("=== Godunov Flash Flood Baseline ===")
    print(f"L = {domain.L/1000:.1f} km, N = {domain.N}, ds = {domain.ds:.3f} m")
    print(f"dt = {time.dt:.3f} s, T = {time.T/60:.1f} min, steps = {time.n_steps}")
    print(f"Stored snapshots: {out['A'].shape[0]} (every 20 steps)\n")

    # 4. Generate Core Profile and Heatmap Plots
    plotting.plot_profiles(
        out,
        times_s=[0.0, 3600.0, 7200.0, 10800.0, 14400.0],  # 0 to 4 hours
        title="Evolution of wetted area profiles $A(s,t)$"
    )
    plotting.plot_spacetime_heatmap(out, title="Space–time evolution of $A(s,t)$")

    # 5. Generate the Flood Hydrograph (Observer standing 30km downstream)
    plotting.plot_hydrograph(
        out,
        target_s=30_000.0,
        title="Flood Hydrograph"
    )

    # 6. Generate the Method of Characteristics Diagram
    # We create starting points P across the initial rain hump
    P_vals = np.linspace(ic.s0 - 3*ic.sigma, ic.s0 + 3*ic.sigma, 40)
    # Get the initial area A at each P
    A_init = gaussian_initial_condition(P_vals, ic=ic)
    # Calculate how fast each point of water is moving
    wave_speeds = wave_speed_rectangular(A_init, phys=phys, w=w_rect)

    plotting.plot_characteristics(
        P_vals, wave_speeds, t_max=time.T, title="Intersecting Characteristics (Proof of Shock)"
    )

    # 7. Compare Numerical Shock to Theoretical Rankine-Hugoniot Trajectory
    plotting.plot_shock_trajectory(
        out, phys=phys, ic=ic, w_rect=w_rect, title="Validation: Shock Trajectory Analysis"
    )

if __name__ == "__main__":
    main()