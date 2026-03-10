"""
plotting.py

Matplotlib utilities for visualizing the 1D flash flood Godunov simulation.
Includes hydrographs and characteristic diagrams for report generation.
"""

from typing import Dict, Sequence, Optional
import numpy as np
import matplotlib.pyplot as plt

def _nearest_index(arr: np.ndarray, target: float) -> int:
    """Get the index k such that arr[k] is closest to target."""
    arr = np.asarray(arr, dtype=float)
    return int(np.argmin(np.abs(arr - float(target))))

def plot_profiles(
    out: Dict[str, np.ndarray],
    *,
    times_s: Sequence[float],
    use_km: bool = True,
    title: str = "Evolution of $A(s,t)$ profiles",
    show: bool = True,
    savepath: Optional[str] = None
) -> plt.Figure:
    """Plot A(s,t) across distance s at several selected times."""
    s, t, A_hist = out["s"], out["t"], out["A"]
    x = s / 1000.0 if use_km else s
    xlab = "Distance downstream $s$ (km)" if use_km else "Distance downstream $s$ (m)"

    fig, ax = plt.subplots()
    for tt in times_s:
        k = _nearest_index(t, tt)
        ax.plot(x, A_hist[k], label=f"$t$ = {t[k]/60:.0f} min")

    ax.set_xlabel(xlab)
    ax.set_ylabel("Wetted area $A$ (m$^2$)")
    ax.set_title(title)
    ax.legend()

    if savepath: fig.savefig(savepath, dpi=200, bbox_inches="tight")
    if show: plt.show()
    return fig

def plot_spacetime_heatmap(
    out: Dict[str, np.ndarray],
    *,
    use_km: bool = True,
    use_minutes: bool = True,
    title: str = "Space–time evolution of $A(s,t)$",
    show: bool = True,
    savepath: Optional[str] = None
) -> plt.Figure:
    """Plot a space–time heatmap of A(s,t)."""
    s, t, A_hist = out["s"], out["t"], out["A"]
    x = s / 1000.0 if use_km else s
    y = t / 60.0 if use_minutes else t
    xlab = "Distance downstream $s$ (km)" if use_km else "Distance downstream $s$ (m)"
    ylab = "Time $t$ (min)" if use_minutes else "Time $t$ (s)"

    fig, ax = plt.subplots()
    im = ax.imshow(
        A_hist, aspect="auto", origin="lower", extent=[x[0], x[-1], y[0], y[-1]]
    )

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="$A$ (m$^2$)")

    if savepath: fig.savefig(savepath, dpi=200, bbox_inches="tight")
    if show: plt.show()
    return fig

def plot_hydrograph(
    out: Dict[str, np.ndarray],
    *,
    target_s: float,
    use_km: bool = True,
    use_minutes: bool = True,
    title: str = "Flood Hydrograph",
    show: bool = True,
    savepath: Optional[str] = None
) -> plt.Figure:
    """Plot a hydrograph A(t) at a specific, fixed location s."""
    s, t, A_hist = out["s"], out["t"], out["A"]
    idx = _nearest_index(s, target_s)
    actual_s = s[idx]

    A_time = A_hist[:, idx] # Extract the column for this specific location

    y = t / 60.0 if use_minutes else t
    ylab = "Time $t$ (min)" if use_minutes else "Time $t$ (s)"
    loc_str = f"{actual_s/1000.0:.1f} km" if use_km else f"{actual_s:.0f} m"

    fig, ax = plt.subplots()
    ax.plot(y, A_time, color='darkred', linewidth=2)
    ax.fill_between(y, A_time, A_time.min(), color='red', alpha=0.1)

    ax.set_xlabel(ylab)
    ax.set_ylabel("Wetted area $A$ (m$^2$)")
    ax.set_title(f"{title} at $s \u2500$ {loc_str}")

    if savepath: fig.savefig(savepath, dpi=200, bbox_inches="tight")
    if show: plt.show()
    return fig


def plot_characteristics(
        P_vals: np.ndarray,
        wave_speeds: np.ndarray,
        t_max: float,
        t_star: Optional[float] = None,
        t_shock: Optional[np.ndarray] = None,
        s_shock: Optional[np.ndarray] = None,
        use_km: bool = True,
        use_minutes: bool = True,
        title: str = "Characteristic Curves & Shock Trajectory",
        show: bool = True,
        savepath: Optional[str] = None
) -> plt.Figure:
    """Plot theoretical characteristic lines, overlaying the exact breaking time and shock path."""
    fig, ax = plt.subplots(figsize=(8, 6))
    t_vals = np.array([0, t_max])

    # 1. Plot the Characteristic Fan (The background)
    for P, v in zip(P_vals, wave_speeds):
        s_vals = P + v * t_vals  # x(t) = P + v*t

        x_plot = s_vals / 1000.0 if use_km else s_vals
        y_plot = t_vals / 60.0 if use_minutes else t_vals
        ax.plot(x_plot, y_plot, color='black', alpha=0.3, linewidth=0.8)

    # 2. Plot the Wave Breaking Time (t*)
    if t_star is not None and np.isfinite(t_star) and t_star <= t_max:
        y_star = t_star / 60.0 if use_minutes else t_star
        ax.axhline(y_star, color='blue', linestyle=':', linewidth=1.5,
                   label=f"Gradient Catastrophe ($t^*$ = {y_star:.1f} min)")

        # 3. Plot the Shock Trajectory (The thick red line)
        if t_shock is not None and s_shock is not None and t_star is not None:
            # We only plot the shock AFTER it physically forms at t*
            mask = t_shock >= t_star
            if np.any(mask):
                x_shock_plot = s_shock[mask] / 1000.0 if use_km else s_shock[mask]
                y_shock_plot = t_shock[mask] / 60.0 if use_minutes else t_shock[mask]

                ax.plot(x_shock_plot, y_shock_plot, color='red', linewidth=3, label="Propagating Shock Front")
                ax.plot(x_shock_plot[0], y_shock_plot[0], 'ro', markersize=6)  # The exact birth point

                # --- NEW: Explicitly label the exact coordinate ---
                s_label = f"{x_shock_plot[0]:.1f} km" if use_km else f"{x_shock_plot[0]:.0f} m"
                t_label = f"{y_shock_plot[0]:.1f} min" if use_minutes else f"{y_shock_plot[0]:.0f} s"

                ax.annotate(
                    f"Shock Formation\n$s \approx$ {s_label}\n$t \approx$ {t_label}",
                    xy=(x_shock_plot[0], y_shock_plot[0]),
                    xytext=(x_shock_plot[0] + 2, y_shock_plot[0] - 25),  # Offsets the text box
                    arrowprops=dict(facecolor='black', arrowstyle='->'),
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", alpha=0.9)
                )
                # ------------------------------------------------

    xlab = "Distance downstream $s$ (km)" if use_km else "Distance downstream $s$ (m)"
    ylab = "Time $t$ (min)" if use_minutes else "Time $t$ (s)"

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)

    x_max = (P_vals[-1] + wave_speeds[-1] * t_max)
    ax.set_xlim([0, x_max / 1000.0 if use_km else x_max])
    ax.set_ylim([0, t_max / 60.0 if use_minutes else t_max])

    # Only show the legend if we actually passed the shock data
    if t_star is not None:
        ax.legend(loc="lower right")

    if savepath: fig.savefig(savepath, dpi=200, bbox_inches="tight")
    if show: plt.show()
    return fig


def plot_shock_trajectory(
        out: Dict[str, np.ndarray],
        phys: 'PhysicalParams',
        ic: 'InitialConditionParams',
        w_rect: float,
        use_km: bool = True,
        use_minutes: bool = True,
        title: str = "Shock Trajectory: Numerical vs Rankine-Hugoniot",
        show: bool = True,
        savepath: Optional[str] = None
) -> plt.Figure:
    """Compare numerical shock with R-H theory, starting precisely at t*."""
    t, s, A_hist = out["t"], out["s"], out["A"]

    s_num = np.zeros_like(t)
    s_RH = np.zeros_like(t)

    s_num[0] = ic.s0
    s_RH[0] = ic.s0

    def Q_rect(A_val):
        return phys.K * (A_val ** 1.5) / np.sqrt(w_rect)

    # --- NEW: Calculate exact breaking time t* ---
    # 1. Get initial area and analytical wave speed
    A0 = A_hist[0]
    l_A0 = w_rect + (2.0 * A0) / w_rect
    term1 = 1.5 * np.sqrt(A0) / np.sqrt(l_A0)
    term2 = (A0 ** 1.5) / (w_rect * (l_A0 ** 1.5))
    v0 = phys.K * (term1 - term2)

    # 2. Find the steepest negative velocity gradient
    dv0_ds = np.gradient(v0, s)
    min_grad = np.min(dv0_ds)

    if min_grad < 0:
        t_star = -1.0 / min_grad
    else:
        t_star = np.inf  # Wave never breaks
    # ---------------------------------------------

    for k in range(1, len(t)):
        dt = t[k] - t[k - 1]

        # 1. Numerical Shock (Steepest gradient in Godunov output)
        grad = np.diff(A_hist[k])
        idx_shock = int(np.argmin(grad))
        s_num[k] = s[idx_shock]

        # 2. Theoretical R-H Speed
        A_L = np.max(A_hist[k])
        A_R = ic.A_base

        if A_L > A_R:
            v_RH = (Q_rect(A_L) - Q_rect(A_R)) / (A_L - A_R)
        else:
            v_RH = 0.0

        # 3. THE FIX: Synchronize before t*, integrate after t*
        if t[k] <= t_star:
            s_RH[k] = s_num[k]  # Lock theory to reality while wave is continuous
        else:
            s_RH[k] = s_RH[k - 1] + v_RH * dt  # Integrate R-H jump condition

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))

    y_plot = t / 60.0 if use_minutes else t
    x_num = s_num / 1000.0 if use_km else s_num
    x_RH = s_RH / 1000.0 if use_km else s_RH

    # Plot numerical tracker normally
    ax.plot(x_num, y_plot, color='blue', linewidth=3, label="Godunov Numerical Tracker")

    # --- NEW: Only plot the R-H theory AFTER the wave actually breaks ---
    mask = t >= t_star
    if np.any(mask):
        ax.plot(x_RH[mask], y_plot[mask], color='red', linestyle='--', linewidth=2,
                label=f"Rankine-Hugoniot Theory (Starts at $t^*$ = {t_star / 60.0:.1f} min)")

        # Plot a dot at the exact creation point of the shock
        first_idx = np.argmax(mask)
        ax.plot(x_RH[first_idx], y_plot[first_idx], 'ro', markersize=6, label="Shock Formation Point")

    xlab = "Distance downstream $s$ (km)" if use_km else "Distance downstream $s$ (m)"
    ylab = "Time $t$ (min)" if use_minutes else "Time $t$ (s)"

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.legend(loc="upper left")

    if savepath: fig.savefig(savepath, dpi=200, bbox_inches="tight")
    if show: plt.show()
    return fig