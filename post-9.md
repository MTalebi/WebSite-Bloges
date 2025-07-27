---
title: "Kalman Filtering for State and Input-State Identification in Linear Systems"
date: "2025-01-27"
description: "Comprehensive guide to Kalman filtering techniques for state estimation and input-state identification in linear dynamic systems. Includes theoretical foundations, practical implementations, and interactive demonstrations."
author: "M. Talebi-Kalaleh"
tags: ["kalman filter", "state estimation", "system identification", "linear systems", "optimal filtering", "control theory", "signal processing"]
category: "engineering"
readTime: "15 min read"
---

# Introduction

In the realm of dynamic systems, engineers and researchers often face the challenge of estimating unknown states and inputs from noisy measurements. The Kalman filter, developed by Rudolf E. Kalman in 1960, provides an elegant and optimal solution to this fundamental problem. This powerful algorithm has become the cornerstone of modern state estimation, finding applications across aerospace, robotics, signal processing, and countless other engineering domains.

This post explores Kalman filtering techniques specifically tailored for linear systems, with a focus on two critical problems:
1. **State Estimation**: Reconstructing the complete state vector from partial, noisy observations
2. **Input-State Identification**: Simultaneously estimating both unknown inputs and system states

We'll develop the theoretical foundations, provide practical implementation guidelines, and demonstrate the concepts through interactive examples that showcase the filter's remarkable capabilities.

---

## The Kalman Filter: Mathematical Foundation

The Kalman filter operates on linear dynamic systems described by the following state-space model:

$$
\begin{align}
\mathbf{x}_{k+1} &= \mathbf{A}_k \mathbf{x}_k + \mathbf{B}_k \mathbf{u}_k + \mathbf{w}_k \tag{1} \\
\mathbf{y}_k &= \mathbf{C}_k \mathbf{x}_k + \mathbf{D}_k \mathbf{u}_k + \mathbf{v}_k \tag{2}
\end{align}
$$

Where:
- **$\mathbf{x}_k \in \mathbb{R}^n$**: State vector at time step $k$
- **$\mathbf{u}_k \in \mathbb{R}^m$**: Known input vector
- **$\mathbf{y}_k \in \mathbb{R}^p$**: Measurement vector
- **$\mathbf{A}_k$**: State transition matrix
- **$\mathbf{B}_k$**: Input matrix
- **$\mathbf{C}_k$**: Output matrix
- **$\mathbf{D}_k$**: Feedthrough matrix
- **$\mathbf{w}_k \sim \mathcal{N}(0, \mathbf{Q}_k)$**: Process noise (Gaussian)
- **$\mathbf{v}_k \sim \mathcal{N}(0, \mathbf{R}_k)$**: Measurement noise (Gaussian)

The Kalman filter provides the optimal estimate $\hat{\mathbf{x}}_k$ that minimizes the mean squared error:

$$
J = \mathbb{E}\left[(\mathbf{x}_k - \hat{\mathbf{x}}_k)^T(\mathbf{x}_k - \hat{\mathbf{x}}_k)\right]
$$

```svg
<svg width="800" height="500" viewBox="0 0 800 500" xmlns="http://www.w3.org/2000/svg">
  <!-- Background -->
  <defs>
    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#f8fafc;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#e2e8f0;stop-opacity:1" />
    </linearGradient>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="3" dy="5" stdDeviation="4" flood-opacity="0.15"/>
    </filter>
    <marker id="arrow" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#374151"/>
    </marker>
    <linearGradient id="boxGrad1" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#3b82f6;stop-opacity:0.1" />
      <stop offset="100%" style="stop-color:#1e40af;stop-opacity:0.2" />
    </linearGradient>
    <linearGradient id="boxGrad2" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#059669;stop-opacity:0.1" />
      <stop offset="100%" style="stop-color:#047857;stop-opacity:0.2" />
    </linearGradient>
    <linearGradient id="boxGrad3" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#dc2626;stop-opacity:0.1" />
      <stop offset="100%" style="stop-color:#b91c1c;stop-opacity:0.2" />
    </linearGradient>
  </defs>
  
  <rect width="800" height="500" fill="url(#bgGrad)"/>
  
  <!-- Title -->
  <text x="400" y="30" text-anchor="middle" font-size="20" font-weight="bold" fill="#1e293b" font-family="Arial">
    Kalman Filter Block Diagram
  </text>
  
  <!-- System Dynamics Block -->
  <rect x="50" y="80" width="120" height="60" rx="8" fill="url(#boxGrad1)" stroke="#1e40af" 
        stroke-width="2" filter="url(#shadow)"/>
  <text x="110" y="105" text-anchor="middle" font-size="12" font-weight="bold" fill="#1e40af" font-family="Arial">
    System Dynamics
  </text>
  <text x="110" y="120" text-anchor="middle" font-size="10" fill="#1e40af" font-family="Arial">
    x_{k+1} = Ax_k + Bu_k + w_k
  </text>
  
  <!-- Measurement Block -->
  <rect x="50" y="200" width="120" height="60" rx="8" fill="url(#boxGrad2)" stroke="#047857" 
        stroke-width="2" filter="url(#shadow)"/>
  <text x="110" y="225" text-anchor="middle" font-size="12" font-weight="bold" fill="#047857" font-family="Arial">
    Measurement
  </text>
  <text x="110" y="240" text-anchor="middle" font-size="10" fill="#047857" font-family="Arial">
    y_k = Cx_k + Du_k + v_k
  </text>
  
  <!-- Kalman Filter Block -->
  <rect x="300" y="140" width="120" height="60" rx="8" fill="url(#boxGrad3)" stroke="#b91c1c" 
        stroke-width="2" filter="url(#shadow)"/>
  <text x="360" y="155" text-anchor="middle" font-size="12" font-weight="bold" fill="#b91c1c" font-family="Arial">
    Kalman Filter
  </text>
  <text x="360" y="170" text-anchor="middle" font-size="10" fill="#b91c1c" font-family="Arial">
    Predict & Update
  </text>
  <text x="360" y="185" text-anchor="middle" font-size="10" fill="#b91c1c" font-family="Arial">
    x̂_k, P_k
  </text>
  
  <!-- State Estimate Output -->
  <rect x="500" y="140" width="120" height="60" rx="8" fill="#f3f4f6" stroke="#6b7280" 
        stroke-width="2" filter="url(#shadow)"/>
  <text x="560" y="165" text-anchor="middle" font-size="12" font-weight="bold" fill="#374151" font-family="Arial">
    State Estimate
  </text>
  <text x="560" y="180" text-anchor="middle" font-size="10" fill="#6b7280" font-family="Arial">
    x̂_k
  </text>
  
  <!-- Arrows -->
  <line x1="170" y1="110" x2="300" y2="170" stroke="#374151" stroke-width="2" marker-end="url(#arrow)"/>
  <line x1="170" y1="230" x2="300" y2="170" stroke="#374151" stroke-width="2" marker-end="url(#arrow)"/>
  <line x1="420" y1="170" x2="500" y2="170" stroke="#374151" stroke-width="2" marker-end="url(#arrow)"/>
  
  <!-- Labels -->
  <text x="235" y="130" text-anchor="middle" font-size="10" fill="#6b7280" font-family="Arial">u_k</text>
  <text x="235" y="250" text-anchor="middle" font-size="10" fill="#6b7280" font-family="Arial">y_k</text>
  <text x="460" y="155" text-anchor="middle" font-size="10" fill="#6b7280" font-family="Arial">x̂_k</text>
  
  <!-- Noise Labels -->
  <text x="40" y="115" text-anchor="middle" font-size="9" fill="#dc2626" font-family="Arial">w_k</text>
  <text x="40" y="235" text-anchor="middle" font-size="9" fill="#dc2626" font-family="Arial">v_k</text>
  
  <!-- Legend -->
  <rect x="50" y="320" width="700" height="160" rx="8" fill="white" stroke="#d1d5db" stroke-width="1" opacity="0.9"/>
  <text x="400" y="340" text-anchor="middle" font-size="14" font-weight="bold" fill="#374151" font-family="Arial">
    Kalman Filter Algorithm Components
  </text>
  
  <!-- Prediction Step -->
  <rect x="70" y="350" width="150" height="40" rx="4" fill="#dbeafe" stroke="#3b82f6" stroke-width="1"/>
  <text x="145" y="365" text-anchor="middle" font-size="11" font-weight="bold" fill="#1e40af" font-family="Arial">Prediction Step</text>
  <text x="145" y="380" text-anchor="middle" font-size="9" fill="#1e40af" font-family="Arial">x̂_{k|k-1} = Ax̂_{k-1} + Bu_k</text>
  
  <!-- Update Step -->
  <rect x="240" y="350" width="150" height="40" rx="4" fill="#dcfce7" stroke="#059669" stroke-width="1"/>
  <text x="315" y="365" text-anchor="middle" font-size="11" font-weight="bold" fill="#047857" font-family="Arial">Update Step</text>
  <text x="315" y="380" text-anchor="middle" font-size="9" fill="#047857" font-family="Arial">x̂_k = x̂_{k|k-1} + K_k(y_k - Cx̂_{k|k-1})</text>
  
  <!-- Gain Calculation -->
  <rect x="410" y="350" width="150" height="40" rx="4" fill="#fef2f2" stroke="#dc2626" stroke-width="1"/>
  <text x="485" y="365" text-anchor="middle" font-size="11" font-weight="bold" fill="#b91c1c" font-family="Arial">Gain Calculation</text>
  <text x="485" y="380" text-anchor="middle" font-size="9" fill="#b91c1c" font-family="Arial">K_k = P_{k|k-1}C^T(CP_{k|k-1}C^T + R)^{-1}</text>
  
  <!-- Covariance Update -->
  <rect x="580" y="350" width="150" height="40" rx="4" fill="#f3e8ff" stroke="#7c3aed" stroke-width="1"/>
  <text x="655" y="365" text-anchor="middle" font-size="11" font-weight="bold" fill="#5b21b6" font-family="Arial">Covariance Update</text>
  <text x="655" y="380" text-anchor="middle" font-size="9" fill="#5b21b6" font-family="Arial">P_k = (I - K_kC)P_{k|k-1}</text>
  
  <!-- Bottom Row -->
  <rect x="70" y="410" width="150" height="40" rx="4" fill="#fef3c7" stroke="#f59e0b" stroke-width="1"/>
  <text x="145" y="425" text-anchor="middle" font-size="11" font-weight="bold" fill="#d97706" font-family="Arial">Process Noise</text>
  <text x="145" y="440" text-anchor="middle" font-size="9" fill="#d97706" font-family="Arial">w_k ~ N(0, Q_k)</text>
  
  <rect x="240" y="410" width="150" height="40" rx="4" fill="#fce7f3" stroke="#ec4899" stroke-width="1"/>
  <text x="315" y="425" text-anchor="middle" font-size="11" font-weight="bold" fill="#be185d" font-family="Arial">Measurement Noise</text>
  <text x="315" y="440" text-anchor="middle" font-size="9" fill="#be185d" font-family="Arial">v_k ~ N(0, R_k)</text>
  
  <rect x="410" y="410" width="150" height="40" rx="4" fill="#e0f2fe" stroke="#0284c7" stroke-width="1"/>
  <text x="485" y="425" text-anchor="middle" font-size="11" font-weight="bold" fill="#0369a1" font-family="Arial">Optimal Estimate</text>
  <text x="485" y="440" text-anchor="middle" font-size="9" fill="#0369a1" font-family="Arial">Minimizes MSE</text>
  
  <rect x="580" y="410" width="150" height="40" rx="4" fill="#f0fdf4" stroke="#16a34a" stroke-width="1"/>
  <text x="655" y="425" text-anchor="middle" font-size="11" font-weight="bold" fill="#15803d" font-family="Arial">Recursive</text>
  <text x="655" y="440" text-anchor="middle" font-size="9" fill="#15803d" font-family="Arial">Online Processing</text>
</svg>
^[figure-caption]("Kalman filter block diagram showing the relationship between system dynamics, measurements, and the optimal estimation process")

---

## The Standard Kalman Filter Algorithm

The Kalman filter operates in two main phases: **prediction** and **update**. Here's the complete algorithm:

### Prediction Step (Time Update)

$$
\begin{align}
\hat{\mathbf{x}}_{k|k-1} &= \mathbf{A}_{k-1} \hat{\mathbf{x}}_{k-1|k-1} + \mathbf{B}_{k-1} \mathbf{u}_{k-1} \tag{3} \\
\mathbf{P}_{k|k-1} &= \mathbf{A}_{k-1} \mathbf{P}_{k-1|k-1} \mathbf{A}_{k-1}^T + \mathbf{Q}_{k-1} \tag{4}
\end{align}
$$

### Update Step (Measurement Update)

$$
\begin{align}
\mathbf{K}_k &= \mathbf{P}_{k|k-1} \mathbf{C}_k^T (\mathbf{C}_k \mathbf{P}_{k|k-1} \mathbf{C}_k^T + \mathbf{R}_k)^{-1} \tag{5} \\
\hat{\mathbf{x}}_{k|k} &= \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}_k (\mathbf{y}_k - \mathbf{C}_k \hat{\mathbf{x}}_{k|k-1} - \mathbf{D}_k \mathbf{u}_k) \tag{6} \\
\mathbf{P}_{k|k} &= (\mathbf{I} - \mathbf{K}_k \mathbf{C}_k) \mathbf{P}_{k|k-1} \tag{7}
\end{align}
$$

Where:
- **$\hat{\mathbf{x}}_{k|k-1}$**: A priori state estimate (before measurement)
- **$\hat{\mathbf{x}}_{k|k}$**: A posteriori state estimate (after measurement)
- **$\mathbf{P}_{k|k-1}$**: A priori error covariance
- **$\mathbf{P}_{k|k}$**: A posteriori error covariance
- **$\mathbf{K}_k$**: Kalman gain matrix

> 💡 **Key Insight:** The Kalman gain $\mathbf{K}_k$ optimally balances the trade-off between prediction uncertainty and measurement reliability. When measurements are noisy (large $\mathbf{R}_k$), the gain decreases, relying more on the model. When measurements are accurate (small $\mathbf{R}_k$), the gain increases, trusting measurements more.