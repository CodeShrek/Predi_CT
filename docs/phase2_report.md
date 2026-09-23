# Phase 2: Coronary Hemodynamics PINN — Technical Report

**Project:** PrediCT — Physics-Informed Neural Network for Coronary ESS Prediction  
**Phase:** 2 (PINN solver + ESS export)  
**Status:** Final implementation — GSoC presentation build

---

## 1. Pipeline Overview

```
CTA Volume (synthetic)
        │
        ▼
  Vessel Segmentation Mask
        │
        ▼
  Centerline Extraction (Skeletonization + Murray's Law)
        │
        ▼
  Adaptive Collocation Sampling
  (Interior · Wall · Inlet · Outlet)
        │
        ▼
  HemodynamicsPINN (6-layer MLP + Fourier Features + Skip Connections)
        │
        ▼
  Loss Function:
  λ_cont · ‖∇·u*‖² + λ_mom · ‖NS residual‖² +
  λ_wall · ‖u*(wall)‖² + λ_inlet · ‖u*(inlet)−u_parabolic‖² +
  λ_outlet · ‖traction(outlet)‖² + λ_mass · ‖Q_in−Q_out‖²
        │
        ▼
  Velocity & Pressure Fields (u*, v*, w*, p*)
        │
        ▼
  ESS Computation via Viscous Stress Tensor Projection
        │
        ▼
  Export: CSV · VTP · NIfTI · JSON · Figures
```

---

## 2. Governing Equations

### 2.1 Steady Incompressible Navier-Stokes (non-dimensional)

Continuity:
$$\nabla^* \cdot \mathbf{u}^* = 0$$

Momentum:
$$(\mathbf{u}^* \cdot \nabla^*)\,\mathbf{u}^* = -\nabla^* p^* + \frac{1}{\text{Re}}\,\nabla^{*2}\mathbf{u}^*$$

Non-dimensionalisation:
$$x^* = \frac{x}{L_0}, \quad \mathbf{u}^* = \frac{\mathbf{u}}{U_0}, \quad p^* = \frac{p}{\rho U_0^2}$$

Reference scales: $L_0 = 3\,\text{mm}$, $U_0 = 0.25\,\text{m/s}$, $\text{Re} \approx 227$

### 2.2 Boundary Conditions

| Boundary | Condition | Formulation |
|:--|:--|:--|
| Inlet | Parabolic Dirichlet | $u^* = U_{max}\left(1 - \frac{r^{*2}}{R^{*2}}\right)\hat{n}_{in}$ |
| Wall | No-slip Dirichlet | $\mathbf{u}^* = \mathbf{0}$ |
| Outlet | Zero-traction Neumann | $-p^*\mathbf{I}\cdot\hat{n} + \frac{1}{\text{Re}}\nabla\mathbf{u}^*\cdot\hat{n} = \mathbf{0}$ |
| Global | Integral mass balance | $Q_{in} = Q_{out}$ |

### 2.3 Endothelial Shear Stress (ESS)

The wall shear stress vector is obtained from the viscous stress tensor:
$$\boldsymbol{\tau}_w = \tau_{ref}\left[\left(\mathbf{J}^* + \mathbf{J}^{*T}\right)\cdot\hat{n}_w - \left(\hat{n}_w^T \cdot (\mathbf{J}^* + \mathbf{J}^{*T}) \cdot \hat{n}_w\right)\hat{n}_w\right]$$

where $\mathbf{J}^* = \partial \mathbf{u}^* / \partial \mathbf{x}^*$ and $\tau_{ref} = \mu U_0 / L_0 \approx 0.292\,\text{Pa}$.

The ESS magnitude is $\text{ESS} = \|\boldsymbol{\tau}_w\|_2$.

---

## 3. Architecture

| Component | Specification |
|:--|:--|
| Input | 3D coordinates $(x^*, y^*, z^*)$ |
| Fourier embedding | 32 frequencies, $\sigma = 2.0$ |
| Hidden layers | 6 × 128 neurons |
| Activation | SiLU (Swish) |
| Skip connections | Every 2 layers |
| Output | $(u^*, v^*, w^*, p^*)$ |
| Total parameters | ~170k |
| Optimizer | Adam, lr = 1e-3 |
| Epochs | 5,000 |

---

## 4. Loss Function

$$\mathcal{L}_{total} = \lambda_c \mathcal{L}_{mass} + \lambda_m \mathcal{L}_{mom} + \lambda_w \mathcal{L}_{wall} + \lambda_{in} \mathcal{L}_{inlet} + \lambda_{out} \mathcal{L}_{outlet} + \lambda_Q \mathcal{L}_{mass\_flux}$$

| Term | Weight | Purpose |
|:--|:--|:--|
| $\mathcal{L}_{mass}$ = $\text{MSE}(\nabla\cdot\mathbf{u}^*)$ | $\lambda_c = 10$ | Local incompressibility |
| $\mathcal{L}_{mom}$ = $\text{MSE}(\text{NS residual})$ | $\lambda_m = 1$ | Momentum conservation |
| $\mathcal{L}_{wall}$ = $\text{MSE}(\mathbf{u}^*_{wall})$ | $\lambda_w = 10$ | No-slip enforcement |
| $\mathcal{L}_{inlet}$ = $\text{MSE}(\mathbf{u}^*_{inlet} - \mathbf{u}_{parabolic})$ | $\lambda_{in} = 10$ | Inlet profile |
| $\mathcal{L}_{outlet}$ = $\text{MSE}(\text{traction})$ | $\lambda_{out} = 1$ | Outflow condition |
| $\mathcal{L}_{mass\_flux}$ = $(Q_{in} - Q_{out})^2$ | $\lambda_Q = 1$ | Global conservation |

---

## 5. Collocation Sampling

| Region | Points | Strategy |
|:--|:--|:--|
| Interior | 4,000 | Adaptive (stenosis/bifurcation weighted) |
| Wall | 2,000 | Surface-weighted |
| Inlet | 500 | Uniform disk |
| Outlet | 500 | Uniform disk |

Adaptive Residual Refinement (RAR) runs at epoch 5,000, replacing 1,000 lowest-residual interior points with the 1,000 highest-residual candidates from a 10,000-point pool.

---

## 6. Validation Summary

| Metric | Value |
|:--|:--|
| Inlet velocity RMSE | ~0.044 (non-dim) |
| Mass conservation error | ~24.9% |
| Mean ESS (clipped) | ~0.096 Pa |
| Median ESS | ~0.083 Pa |
| Wall normal smoothness | 0.920 |
| Training time | ~6,340 s (~1 hr 45 min) |
| Best total loss | 3.64 × 10⁻² |
| Final total loss | 6.27 × 10⁻² |

---

## 7. Key Findings from Scientific Experiments

### Experiment 1: Integral Mass Conservation Loss (Completed)
- **Finding:** `compute_integral_mass_loss()` is mathematically correct. Gradient propagation verified. Low gradient magnitude is a sign of a nearly-satisfied constraint.
- **Action:** No change to loss formulation.

### Experiment 2: Wall-Normal Smoothing (Completed)
- **Finding:** Smoothing wall normals (σ = 1.0 mm) changes ESS by only +2.8%. Normals are NOT the dominant cause of ESS underprediction.
- **Action:** No change to normal computation.

### Experiment 3: Velocity Field Audit (Completed)
- **Finding:** The inlet velocity is correct (centreline = 2.0 non-dim). However, the velocity collapses to ~1.5% of its value within 3 mm downstream. Cross-sectional flux drops from 1.44 ml/s at inlet to ~0 ml/s in the bulk interior.
- **Root cause:** Inlet Conditioning Failure — the network locally destroys mass after the inlet.

### Experiment 4: λ_continuity A/B Test (Completed)
- **Hypothesis:** Increasing λ_c to 10.0 would prevent mass leakage.
- **Finding:** No effect. Training log reported mass MSE = 0.00185, but freshly-sampled diagnostic points showed mean |∇·u| = 0.65–0.75. The network is overfitting the static collocation grid, hiding all mass leakage between training points.
- **Conclusion:** The root cause is **collocation overfitting** on the static interior point grid.

---

## 8. Known Limitations

1. **Static Collocation Grid:** The 4,000 interior points are sampled once before training. The network memorises these coordinates and satisfies the PDE only at those exact locations, while violating it in the physical space between points. This is the primary unresolved limitation.

2. **ESS Underprediction:** Mean ESS (~0.096 Pa) is ~16.5% of the Poiseuille theory value (~0.583 Pa), directly caused by the weak interior velocity field.

3. **Synthetic Geometry:** The pipeline uses a procedurally-generated coronary bifurcation, not a patient-specific CTA reconstruction. Patient-specific integration requires Phase 3 (active registration to real CT data).

4. **Steady-State Only:** The solver models steady incompressible flow. Real coronary hemodynamics are pulsatile; extending to time-dependent Navier-Stokes is a future milestone.

---

## 9. Future Work

| Priority | Intervention | Expected Impact |
|:--|:--|:--|
| **Critical** | Fast batched collocation resampling (pool of 50k–100k points, in-place buffer update per epoch) | Eliminates memorisation; should resolve ESS underprediction |
| High | Physics-informed curriculum (warm-start from continuity only, then add momentum) | Prevents trivial zero-velocity solution during initialisation |
| High | Patient-specific CTA geometry (Phase 3 integration) | Clinically relevant ESS maps |
| Medium | Pulsatile solver extension | Physiologically accurate time-varying WSS |
| Medium | GradNorm adaptive weight balancing | Automatic loss weight tuning |

---

## 10. Repository Structure

```
phase2_v2/
├── run_phase2.py           # Main training pipeline
├── export_to_slicer.py     # Single-command export tool (NIfTI + figures + JSON)
├── config.py               # All physical/numerical parameters
├── network.py              # HemodynamicsPINN architecture
├── physics.py              # Navier-Stokes residuals
├── losses.py               # Individual loss functions + GradNorm balancer
├── boundary_conditions.py  # BC implementations (inlet, wall, outlet, mass flux)
├── sampling.py             # Adaptive collocation point sampler
├── geometry.py             # Centerline extraction + domain generation
├── trainer.py              # PINNTrainer with Adam + L-BFGS + RAR
├── ess.py                  # ESS/WSS computation via Jacobian projection
├── ess_diagnostic.py       # ESS statistical diagnostics and plots
├── validation.py           # Mass conservation + field verification
└── visualization.py        # CSV → VTP export (ParaView / 3D Slicer)
```

---

## 11. 3D Slicer Visualization Guide

### Loading NIfTI outputs

1. Open 3D Slicer
2. `File → Add Data → Choose File(s)` → select from `output_v2/exports/nifti/`
3. Recommended load order:
   - `lumen_mask.nii.gz` → Segment Editor → label map
   - `velocity_magnitude.nii.gz` → Volume Rendering → colour by value
   - `pressure.nii.gz` → Volumes module → Window/Level adjust
   - `ess.nii.gz` → Volume Rendering → hot colormap (atherogenic = 0–1 Pa)

### ESS spatial map (VTP)
- Load `ess_predictions.vtp` via `File → Add Data`
- In `Models` module, set Active Scalar to `ESS`
- Use Jet or Plasma colormap, range 0–0.5 Pa

### Figures
All 6 publication figures are in `output_v2/exports/figures/` at 300 dpi.
