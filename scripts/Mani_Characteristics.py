#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 11:44:28 2026

@author: manigill
"""

import numpy as np
import matplotlib.pyplot as plt


###Constants###

g = 9.81
alpha = 2*2*np.pi/360
f = 0.02
theta = 60*2*np.pi/360

s_min, s_max = -10.0, 30
N_chars = 50

s0 = np.linspace(s_min, s_max, N_chars)



sinAlpha = np.sin(alpha)
sinTheta = np.sin(theta)

# v = C*A
C = (((5/8)*(g*sinAlpha/f)**0.5)*(sinTheta)**0.25) / (2**0.75)


def v_wedge(A):
    A = np.array(A)
    if np.any(A <= 0):
        raise ValueError('A must be positive')
    return C*A**0.25


def dv_wedge_dA(A):
    A = np.array(A)
    if np.any(A <= 0):
        raise ValueError('A must be positive')
    return (C/4) * A**(-0.75)


###IC - Gaussian waveform

def A0_gaussian(s, Ab=1.0, Ap=1.0, sc=0.0, sigma=2.0):
    return Ab + Ap * np.exp(-0.5 * ((s - sc) / sigma)**2)

A0 = A0_gaussian(s0)

v0 = v_wedge(A0)

L = s_max - s_min
t_max = 0.6 * L / np.max(v0)
Nt = 300
t = np.linspace(0.0, t_max, Nt)



# s(t) = s0 + v(A0(s0))*t
S = s0[:, None] + v0[:, None] * t[None, :]






dA0_ds0 = np.gradient(A0, s0)
bracket = dv_wedge_dA(A0) * dA0_ds0

neg = bracket < 0
if np.any(neg):
    t_star = -1.0 / np.min(bracket[neg])  # since min over negative is most negative
else:
    t_star = np.inf  # no predicted crossing for this A0 on this interval


fig = plt.figure(figsize=(10, 4))


ax1 = plt.subplot(1, 2, 1)
ax1.plot(s0, A0)
ax1.set_xlabel("s")
ax1.set_ylabel("A0(s)")
ax1.set_title("Initial condition")


ax2 = plt.subplot(1, 2, 2)
for i in range(N_chars):
    ax2.plot(S[i, :], t, linewidth=0.8)

ax2.set_xlabel("s")
ax2.set_ylabel("t")
ax2.set_title("Characteristics: s(t) = s0 + v(A0(s0)) t")
ax2.set_xlim(s_min, s_max + 0.8 * L)  # widen view so you see advection

if np.isfinite(t_star) and (0 < t_star < t_max):
    ax2.axhline(t_star, linestyle="--", linewidth=1.2)
    ax2.text(0.02, 0.95, f"t* ≈ {t_star:.3g}", transform=ax2.transAxes,
             va="top", ha="left")
elif np.isfinite(t_star):
    ax2.text(0.02, 0.95, f"t* ≈ {t_star:.3g} (outside window)", transform=ax2.transAxes,
             va="top", ha="left")
else:
    ax2.text(0.02, 0.95, "No t* predicted (bracket never < 0)", transform=ax2.transAxes,
             va="top", ha="left")

plt.tight_layout()
plt.show()


# //==================================================\\
# ||          Analytical Profiles over time           ||
# \\==================================================//

def plot_analytical_profiles(S_matrix, A0_array, t_array, t_star_val, num_profiles=5):
    fig, ax = plt.subplots(figsize=(8, 5))

    # Pick a few evenly spaced time indices to plot
    indices = np.linspace(0, len(t_array) - 1, num_profiles, dtype=int)

    for idx in indices:
        time_val = t_array[idx]

        # S_matrix[:, idx] contains the spatial positions of all points at this time
        ax.plot(S_matrix[:, idx], A0_array, label=f"t = {time_val:.2f}")

    ax.set_xlabel("s")
    ax.set_ylabel("A(s,t)")
    ax.set_title("Evolution of A(s,t) Profiles (Analytical)")

    # Add a warning line for the breaking time
    if np.isfinite(t_star_val):
        ax.axvline(x=S_matrix[np.argmax(A0_array), np.argmin(np.abs(t_array - t_star_val))],
                   color='red', linestyle='--', alpha=0.5, label=f"Breaking Point (t={t_star_val:.2f})")

    ax.legend()
    plt.tight_layout()
    plt.show()


# Call the function using his variables
plot_analytical_profiles(S, A0, t, t_star)

