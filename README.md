# 1D Flash Flood Modelling (MATH6149)

This repository contains a numerical simulation of flash flood formation in desert canyons, developed for the MATH6149 – Modelling with Differential Equations module.

The model demonstrates how a smooth upstream rainfall event can evolve into a sudden downstream “wall of water”, a phenomenon commonly observed in desert flash floods. Mathematically, this behaviour corresponds to the formation of a shock wave in a nonlinear conservation law.

## 📖 Physics & Mathematical Background

Directly simulating flood dynamics using the full 3D Navier–Stokes equations is computationally expensive and unnecessary for capturing the essential behaviour. Instead, we use a 1D quasi-steady kinematic wave model, which retains the dominant physics governing flood propagation.

### Governing Equation
The model is based on conservation of mass:

$$A_t + (Q(A))_s = 0$$

where:
* $A(s,t)$ — wetted cross-sectional area
* $Q(A)$ — volumetric discharge (flux)
* $s$ — downstream distance
* $t$ — time

### Turbulent Flow Approximation
Balancing gravitational acceleration with turbulent skin friction (while neglecting inertia) yields an expression for the mean flow velocity:

$$\overline{u} = \sqrt{\frac{g \sin\alpha}{f}\frac{A}{l(A)}}$$

where:
* $g$ — gravitational acceleration
* $\alpha$ — canyon slope angle
* $f$ — turbulent friction factor
* $l(A)$ — wetted perimeter determined by canyon geometry

The volumetric flux is therefore:

$$Q(A) = A \overline{u}$$

### Nonlinear Wave Steepening
For realistic canyon geometries (rectangular, V-shaped, parabolic), the flux function is convex:

$$Q''(A) > 0$$

This implies that the wave speed:

$$v(A) = Q'(A)$$

increases with water depth. Consequently:
* Deeper water at the crest travels faster.
* Shallower water ahead travels slower.

This causes the wave to steepen over time, eventually forming a shock (flood front) — the mathematical explanation for the sudden appearance of flash floods far downstream.

## 🗂️ Project Structure

The codebase is designed to be modular and easy to extend, separating physical modelling, numerical methods, and visualization.

```text
project/
│
├── params.py
│   Configuration via dataclasses:
│   domain size, timestep, physical constants, and initial conditions
│
├── physics.py
│   Core mathematical models:
│   - canyon geometry l(A)
│   - flux function Q(A)
│   - wave speed Q'(A)
│   - initial rainfall pulse
│
├── solver.py
│   Implementation of a 1st-order Godunov finite volume scheme
│   used to solve the conservation law
│
├── plotting.py
│   Matplotlib utilities for generating diagnostic and report figures
│
└── main.py
    Entry point that runs the simulation and produces visual outputs
```

## 🚀 Installation & Usage

### 1. Prerequisites
Ensure Python 3.8+ is installed.

### 2. Install Dependencies
Navigate to the project directory and install the required packages:

```bash
pip install -r requirements.txt
```
### 3. Run the Simulation
```bash
python main.py
```
The solver will simulate **4 hours of flood evolution across a 50 km canyon** and generate diagnostic plots automatically.

---

# 📊 Visual Outputs

Running `main.py` produces **five key diagnostic figures** that illustrate the physics of flash flood formation.

### Evolution of \(A(s,t)\) Profiles
Shows the spatial evolution of the flood wave over time.  
The front of the wave steepens while the rear spreads into a **rarefaction tail**.

### Space–Time Heatmap
A global visualization of water depth in the \((s,t)\) domain.  
The flood front appears as a **diagonal ridge propagating downstream**.

### Flood Hydrograph (at \(s = 30\) km)
Displays water level versus time for a fixed downstream observer.  
This plot clearly shows the **sudden arrival of a flood wall**, despite smooth upstream forcing.

### Intersecting Characteristics
A characteristic diagram in the \((s,t)\) plane illustrating how trajectories corresponding to deeper water intersect those ahead of them.  
This intersection marks the **breakdown of the classical solution** and the formation of a shock.
These calculations have nothing to do with the Godunov method, they are analytical and are there to support the validity of the Godunov simulation.

### Shock Trajectory Validation
Compares the numerically computed shock position with the theoretical propagation speed given by the **Rankine–Hugoniot jump condition**, demonstrating excellent agreement and validating the Godunov scheme.
Numerically, the shock is found by looking for the most negative value of $(\frac{\partial A}{\partial s})$. This will reflect the most negative slope of A and should approximate the location of the shock well.
Analytically, we integrate the Rankine-Hugoniot condition by using the steepest peak of A at the given time for A_l and A_base = 5m^2 for A_r.
---

# 🔬 Numerical Method

The governing conservation law is solved using a **first-order Godunov finite volume method**.

This scheme:

- Preserves the **integral conservation law**
- Correctly captures **shock propagation**
- Is stable under the **CFL condition**

Although first-order methods introduce some **numerical diffusion**, they reliably reproduce the **correct shock speed**.

---

# 📚 Educational Purpose

This project illustrates several important concepts in **nonlinear PDEs and numerical modelling**:

- Nonlinear conservation laws
- Method of characteristics
- Shock formation
- Rankine–Hugoniot jump conditions
- Godunov finite volume methods

It serves as a practical example of how relatively simple models can capture **complex real-world phenomena such as flash floods**.