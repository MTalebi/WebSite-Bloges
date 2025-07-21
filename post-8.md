---
title: "Modeling Dynamic Systems with Parametric State-Space Models"
date: "2025-07-21"
description: "A deep, technical, and accessible guide to modeling physical systems using state-space methods, with equations, SVGs, code, and an interactive app."
author: "M. Talebi"
tags: ["state-space", "differential equations", "mechanical systems", "simulation", "vibration", "system identification"]
category: "tutorial"
readTime: "10 min read"
---

# Modeling Dynamic Systems with Parametric State-Space Models

Many physical systems—especially in engineering and science—are governed by differential equations. Modeling these systems efficiently is crucial for analysis, simulation, and control. **State-space (SS) modeling** is a mathematical method that reduces higher-order differential equations to a set of first-order ordinary differential equations (ODEs), making them easier to solve and simulate.

---

## 1. From Physical Laws to State-Space Models

Consider a mechanical vibration system, such as a mass-spring-damper:

$$
M \ddot{x}(t) + C \dot{x}(t) + K x(t) = F(t)
$$

Where:
- $M$ is the mass matrix
- $C$ is the damping matrix
- $K$ is the stiffness matrix
- $x(t)$ is the displacement vector
- $F(t)$ is the external force vector

To analyze or simulate this system, we convert it to **state-space form** by defining the state vector:

$$
\mathbf{z}(t) = \begin{bmatrix} x(t) \\ \dot{x}(t) \end{bmatrix}
$$

The system becomes:

$$
\dot{\mathbf{z}}(t) = \mathbf{A} \mathbf{z}(t) + \mathbf{B} F(t)
$$

Where:

$$
\mathbf{A} = \begin{bmatrix} 0 & I \\ -M^{-1}K & -M^{-1}C \end{bmatrix}, \quad \mathbf{B} = \begin{bmatrix} 0 \\ M^{-1} \end{bmatrix}
$$

---

```svg
<svg width="420" height="120" viewBox="0 0 420 120" xmlns="http://www.w3.org/2000/svg">
  <rect x="20" y="60" width="60" height="30" fill="#e0e0e0" stroke="#333" stroke-width="2"/>
  <text x="50" y="80" font-size="14" text-anchor="middle" fill="#333">m</text>
  <rect x="80" y="70" width="40" height="10" fill="#b3c6e7" stroke="#333" stroke-width="1"/>
  <text x="100" y="65" font-size="12" text-anchor="middle" fill="#333">k</text>
  <rect x="120" y="70" width="20" height="10" fill="#c2e0c6" stroke="#333" stroke-width="1"/>
  <text x="130" y="65" font-size="12" text-anchor="middle" fill="#333">c</text>
  <line x1="140" y1="75" x2="200" y2="75" stroke="#333" stroke-width="2" marker-end="url(#arrow)"/>
  <text x="170" y="65" font-size="12" text-anchor="middle" fill="#333">F(t)</text>
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="10" refY="5" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L10,5 L0,10 L2,5 z" fill="#333" />
    </marker>
  </defs>
</svg>
```
^[figure-caption]("A mass-spring-damper system: the classic example for state-space modeling.")

---

## 2. Discretizing the State-Space Model

For digital simulation or control, we discretize the continuous model:

$$
\mathbf{z}_{k+1} = \mathbf{A}_d \mathbf{z}_k + \mathbf{B}_d F_k
$$

A simple (Euler) discretization gives:

$$
\mathbf{A}_d = I + \mathbf{A} \Delta t, \quad \mathbf{B}_d = \mathbf{B} \Delta t
$$

---

```svg
<svg width="350" height="80" viewBox="0 0 350 80" xmlns="http://www.w3.org/2000/svg">
  <rect x="10" y="30" width="60" height="20" fill="#e0e0e0" stroke="#333" stroke-width="2"/>
  <text x="40" y="45" font-size="14" text-anchor="middle" fill="#333">z[k]</text>
  <polygon points="70,40 90,30 90,50" fill="#b3c6e7" stroke="#333" stroke-width="1"/>
  <rect x="90" y="30" width="60" height="20" fill="#fff" stroke="#333" stroke-width="2"/>
  <text x="120" y="45" font-size="14" text-anchor="middle" fill="#333">A_d</text>
  <polygon points="150,40 170,30 170,50" fill="#b3c6e7" stroke="#333" stroke-width="1"/>
  <rect x="170" y="30" width="60" height="20" fill="#fff" stroke="#333" stroke-width="2"/>
  <text x="200" y="45" font-size="14" text-anchor="middle" fill="#333">z[k+1]</text>
</svg>
```
^[figure-caption]("Discrete-time state update: $z_{k+1} = A_d z_k + B_d F_k$")

---

## 3. Why State-Space Models?

- **Prediction:** If you know the parameters ($M$, $C$, $K$), you can predict future responses (acceleration, displacement, strain) for any input force.
- **Estimation:** With limited measurements, SS models let you estimate unmeasured states (e.g., velocity from displacement).
- **Foundation for Advanced Methods:** Many identification and prediction algorithms (like the Kalman filter) are built on SS models.
- **Universality:** Most physical systems with differential equations can be modeled in this way.

---

## 4. Implementation: Simulating a Multi-DOF System

Below is a general Python function to simulate a multi-degree-of-freedom (MDOF) system using state-space methods. You provide the mass ($M$), damping ($C$), stiffness ($K$) matrices, and the force vector. The function returns displacement, velocity, and acceleration over time.

```python
import numpy as np

def simulate_state_space(M, C, K, F, dt, x0=None, v0=None):
    """
    Simulate a multi-DOF system using state-space formulation.
    M, C, K: (n, n) arrays (mass, damping, stiffness)
    F: (n, N) array (force for each DOF at each time step)
    dt: time step
    x0, v0: initial displacement and velocity (n,)
    Returns: t, x, v, a (all (n, N))
    """
    n, N = F.shape
    t = np.arange(0, N * dt, dt)
    if x0 is None:
        x0 = np.zeros(n)
    if v0 is None:
        v0 = np.zeros(n)
    # State-space matrices
    A = np.block([
        [np.zeros((n, n)), np.eye(n)],
        [-np.linalg.inv(M) @ K, -np.linalg.inv(M) @ C]
    ])
    B = np.block([
        [np.zeros((n, n))],
        [np.linalg.inv(M)]
    ])
    Ad = np.eye(2 * n) + A * dt
    Bd = B * dt
    # Initialize
    z = np.zeros((2 * n, N))
    z[:n, 0] = x0
    z[n:, 0] = v0
    for k in range(1, N):
        z[:, k] = Ad @ z[:, k - 1] + Bd @ F[:, k - 1]
    x = z[:n, :]
    v = z[n:, :]
    a = -np.linalg.inv(M) @ (K @ x + C @ v) + np.linalg.inv(M) @ F
    return t, x, v, a
```

---

## 5. Interactive: Explore State-Space Simulation

```html
<!-- Minimal interactive app: change M, C, K, and force, see response (placeholder for real JS plot) -->
<div>
  <label>Mass (m): <input type="number" id="mass" value="1" min="0.1" step="0.1"></label>
  <label>Stiffness (k): <input type="number" id="stiffness" value="10" min="0.1" step="0.1"></label>
  <label>Damping (c): <input type="number" id="damping" value="0.5" min="0" step="0.1"></label>
  <button onclick="simulate()">Simulate</button>
  <div id="plot" style="margin-top:1em; height:180px; background:#f8f8f8; border:1px solid #ccc; text-align:center; line-height:180px;">[Plot will appear here]</div>
</div>
<script>
function simulate() {
  const m = parseFloat(document.getElementById('mass').value);
  const k = parseFloat(document.getElementById('stiffness').value);
  const c = parseFloat(document.getElementById('damping').value);
  // Placeholder: In a real app, run a JS simulation and plot
  document.getElementById('plot').innerText = `Simulated response for m=${m}, k=${k}, c=${c}`;
}
</script>
```

---

```svg
<svg width="400" height="100" viewBox="0 0 400 100" xmlns="http://www.w3.org/2000/svg">
  <rect x="20" y="40" width="60" height="20" fill="#e0e0e0" stroke="#333" stroke-width="2"/>
  <text x="50" y="55" font-size="14" text-anchor="middle" fill="#333">States</text>
  <polygon points="80,50 100,40 100,60" fill="#b3c6e7" stroke="#333" stroke-width="1"/>
  <rect x="100" y="40" width="60" height="20" fill="#fff" stroke="#333" stroke-width="2"/>
  <text x="130" y="55" font-size="14" text-anchor="middle" fill="#333">Output</text>
  <polygon points="160,50 180,40 180,60" fill="#b3c6e7" stroke="#333" stroke-width="1"/>
  <rect x="180" y="40" width="60" height="20" fill="#fff" stroke="#333" stroke-width="2"/>
  <text x="210" y="55" font-size="14" text-anchor="middle" fill="#333">Estimate</text>
</svg>
```
^[figure-caption]("State-space variable flow: states, outputs, and estimates.")

---

## 6. Summary

State-space models provide a powerful, universal framework for modeling, simulating, and understanding dynamic systems governed by differential equations. They are especially elegant for vibration and mechanical systems, enabling prediction, estimation, and serving as the foundation for advanced techniques.

> 💡 **Tip:** For more on estimation and filtering, see the upcoming post on the Kalman filter and system identification.

---

## Further Reading
- [State-Space Methods in Control Engineering](https://en.wikipedia.org/wiki/State-space_representation)
- [Mechanical Vibrations: Theory and Applications](https://www.springer.com/gp/book/9783319784445)
- [Kalman Filter (Wikipedia)](https://en.wikipedia.org/wiki/Kalman_filter)
