# Phase 2 Reproducibility Guide

This document outlines the exact environment, configuration, and commands required to reproduce the training and export of the PrediCT Phase 2 Coronary Hemodynamics PINN.

## 1. Environment & Hardware

| Component | Version / Specification |
|:--|:--|
| Python | 3.13.7 |
| PyTorch | 2.13.0 |
| Backend | MPS (Apple Silicon) — Verified `torch.backends.mps.is_available() == True` |
| NumPy | 2.5.1 |
| SciPy | 1.18.0 |
| NiBabel | 5.4.2 |
| Matplotlib | 3.11.1 |
| Pandas | 3.0.5 |
| PyVista | 0.48.4 |

## 2. Configuration & Seed

The pipeline is completely deterministic. The global random seed is injected at the top of the execution script into `torch`, `numpy`, and `random`.

- **Random Seed**: `42`

### Physical & Numerical Configuration
- **Characteristic Length ($L_0$)**: 0.003 m (3 mm)
- **Characteristic Velocity ($U_0$)**: 0.25 m/s
- **Blood Density ($\rho$)**: 1050 kg/m³
- **Dynamic Viscosity ($\mu$)**: 0.0035 Pa·s
- **Reynolds Number ($Re$)**: 227.14
- **Reference Shear Stress ($\tau_{ref}$)**: 0.2917 Pa

### Loss Function Weights
- $\lambda_{continuity}$ = 10.0
- $\lambda_{momentum}$ = 1.0
- $\lambda_{wall\_noslip}$ = 10.0
- $\lambda_{inlet}$ = 10.0
- $\lambda_{outlet}$ = 1.0
- $\lambda_{integral\_mass}$ = 1.0

### Collocation Strategy
- **Interior**: 4,000 points
- **Wall**: 2,000 points
- **Inlet**: 500 points
- **Outlet**: 500 points

## 3. Checkpoint Used for Presentation

The final deliverables were generated using the best model checkpoint from a 19,783-epoch run.

- **Path**: `pinn_checkpoints/best_model.pt`
- **Epoch Saved**: 14,783
- **Best Total Loss**: 7.026 × 10⁻⁴

## 4. Commands

Activate the virtual environment from the repository root:
```bash
source .venv/bin/activate
```

### To Train the Network
This will launch the training loop using Adam (and switch to L-BFGS dynamically if configured) for 5000 epochs by default.
```bash
cd phase2_v2
PYTHONPATH=. python3 run_phase2.py
```

### To Export Results (NIfTI, Figures, JSON)
This uses the saved checkpoint to evaluate the network across the entire 3D voxel grid and the vessel surface, exporting the results for 3D Slicer.
```bash
cd phase2_v2
PYTHONPATH=. python3 export_to_slicer.py
```

## 5. Expected Outputs

Running the export script will generate the following directory structure inside `output_v2/exports/`:

```
output_v2/exports/
├── results_summary.json            # Complete metadata, configuration, and numerical metrics
├── nifti/
│   ├── velocity_magnitude.nii.gz   # 3D speed volume
│   ├── velocity_x.nii.gz           # u-component
│   ├── velocity_y.nii.gz           # v-component
│   ├── velocity_z.nii.gz           # w-component
│   ├── pressure.nii.gz             # Pressure volume
│   ├── ess.nii.gz                  # ESS mapped to wall surface voxels
│   ├── lumen_mask.nii.gz           # Binary mask of the blood flow region
│   ├── wall_mask.nii.gz            # Binary mask of the endothelial wall
│   └── flow_domain.nii.gz          # Discretized domain mask
└── figures/
    ├── phase2_summary.png          # 2x2 presentation slide summarizing key results
    ├── fig_ess_histogram.png
    ├── fig_ess_spatial_map.png
    ├── fig_centreline_velocity.png
    ├── fig_flux_conservation.png
    ├── fig_velocity_histogram.png
    └── fig_pressure_histogram.png
```

## 6. Repository Structure

```
phase2_v2/
├── archive/                  # Archived diagnostic & experimental scripts
├── run_phase2.py             # Main PINN training orchestrator
├── export_to_slicer.py       # Final presentation & Slicer NIfTI exporter
├── config.py                 # Hyperparameter & physical property definitions
├── network.py                # HemodynamicsPINN architecture definition
├── physics.py                # Navier-Stokes residual computations
├── losses.py                 # Multi-objective PINN loss evaluators
├── boundary_conditions.py    # Inlet, Outlet, and No-Slip condition functions
├── sampling.py               # Adaptive point cloud generation
├── geometry.py               # Voxel to skeleton/mesh transformations
├── trainer.py                # Custom training loop with logging & RAR
├── ess.py                    # Endothelial Shear Stress tensor computations
├── ess_diagnostic.py         # Statistical analysis for ESS predictions
├── validation.py             # Conservation and field verification tools
├── visualization.py          # Legacy VTP/CSV generation utilities
├── phase2_report.md          # Technical analysis and findings report
└── reproducibility.md        # This guide
```
