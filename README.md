# PrediCT: Physiologically-Driven Synthetic Coronary Calcium Generation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GSoC 2024](https://img.shields.io/badge/GSoC-2024-orange.svg)](https://summerofcode.withgoogle.com/)

**PrediCT** is an end-to-end computational framework that synthesizes physiologically accurate, anatomically constrained **Coronary Artery Calcium (CAC)** deposits directly into Non-Contrast Chest CT (NCCT) scans.

Rather than relying on unconstrained generative models (GANs/diffusion) that can hallucinate unphysical lesions, PrediCT simulates the actual mechanobiological progression of atherosclerosis: hemodynamic disturbances driving endothelial dysfunction, focal plaque nucleation, and scanner-calibrated partial-volume texturing.

---

## 4-Phase Core Pipeline

| Phase | Module | Method & Function |
|---|---|---|
| **Phase 1** | **Anatomical Scaffolding** | Multi-Atlas Label Propagation (MALP) with NCC pre-selection and Mutual Information voting to extract patient-specific coronary artery trees from NCCT. |
| **Phase 2** | **Hemodynamic Surrogate** | Meshless Physics-Informed Neural Network (PINN) solving 3D incompressible Navier-Stokes equations to compute Endothelial Shear Stress (ESS). |
| **Phase 3** | **Biomechanical Plaque Growth** | Atherogenic thresholding (ESS < 1.0 Pa) and stochastic anisotropic Breadth-First Search (BFS) to grow natural, wall-hugging plaque morphology within strict vessel boundaries. |
| **Phase 4** | **Radiometric Synthesis** | Dynamic Hounsfield Unit (HU) texturing (130–1500 HU) with continuous Gaussian alpha blending to simulate CT partial-volume effects, followed by automated Agatston score validation. |

---

## New Implementations & Features

- **Diffeomorphic B-Spline Registration (`masking_bspline.py`)**: Non-rigid free-form deformation preserving vessel topology across tortuous coronary branches.
- **Interactive 3D Vessel Mesh Visualizer (`view_vessel_3d.py`)**: Real-time PyVista/VTK 3D surface rendering for spatial inspection of vessel geometry, bifurcations, and plaque seeds.
- **Clinical DICOM Conversion Suite (`nifti_to_dicom.py`, `convert_all_patients_dicom.py`)**: Full conversion of synthetic NIfTI volumes into standard 2D DICOM slice series with preserved geometry and metadata for direct loading into clinical PACS and 3D Slicer.
- **Cohort Verification Engine (`run_cohort_verification.py`)**: Automated audit runner validating registration consensus, PINN convergence, and target Agatston accuracy across multi-patient cohorts.
- **Dynamic Environment Configuration (`config.py`)**: Configurable overrides via environment variables for execution across local workstations and GPU clusters.

---

## Repository Structure

```text
PrediCT/
├── src/
│   ├── phase1_segmentation/    # Multi-atlas & B-spline vessel extraction, 3D visualizer
│   ├── phase2_hemodynamics/    # PINN Navier-Stokes solver & ESS calculation
│   ├── phase3_plaque_growth/   # Stochastic seeding & anisotropic BFS growth
│   ├── phase4_texturing/       # Dynamic HU texturing, alpha blending & Agatston score
│   └── utils/                  # Slicer export, VTP generation, visualization
├── scripts/
│   ├── physio_twin.py          # Primary end-to-end pipeline runner
│   ├── physio_twin_bspline.py  # Dedicated B-spline pipeline runner
│   ├── nifti_to_dicom.py       # Clinical DICOM slice exporter
│   └── run_cohort_verification.py # Automated cohort verification
├── requirements.txt            # Python dependencies
└── README.md
```

---

## Quickstart

### 1. Installation
```bash
git clone https://github.com/CodeShrek/Predi_CT.git
cd Predi_CT
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run End-to-End Pipeline
Synthesize calcium deposits on a patient scan to achieve a target Agatston score of 400:
```bash
python scripts/physio_twin.py \
    --patient_dir COCA/1cc17f65f909 \
    --target_agatston 400 \
    --out_dir outputs/
```

### 3. Run with B-Spline Registration
```bash
python scripts/physio_twin_bspline.py \
    --patient_dir COCA/1cc17f65f909 \
    --target_agatston 400 \
    --out_dir outputs/
```

### 4. Inspect Extracted Vessels in 3D
```bash
python src/phase1_segmentation/view_vessel_3d.py \
    --vessel_mask outputs/1cc17f65f909_synthetic_vessel.nii.gz
```

### 5. Convert Synthetic Scan to Clinical DICOM
```bash
python scripts/nifti_to_dicom.py \
    --nifti_path outputs/1cc17f65f909_synthetic_coca.nii.gz \
    --reference_dicom_dir COCA/1cc17f65f909/dicom/ \
    --output_dir outputs/1cc17f65f909_dicom/
```

---

## Applications in Cardiovascular AI

- **Counterfactual Scans**: Synthesize progressive calcium burdens (Agatston 0 -> 100 -> 400 -> 1000) on the same patient anatomy for longitudinal model evaluation.
- **Class Imbalance Resolution**: Balances skewed screening cohorts (>85% zero-calcium) by creating matched synthetic positive scans without patient re-exposure.
- **Shortcut-Free Training**: Plaque obeys physical fluid dynamics and CT radiometric properties, preventing models from learning artificial border shortcuts.

---

## References

1. **Chatzizisis, Y. S., et al. (2007)**. *Role of Endothelial Shear Stress in the Natural History of Coronary Atherosclerosis*. JACC, 49(25), 2379–2393.
2. **Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019)**. *Physics-Informed Neural Networks*. Journal of Computational Physics, 378, 686–707.
3. **Gao, J., et al. (2023)**. *Coronary Calcium and Chest CTs (COCA) Dataset*. PhysioNet.

---

## License
Distributed under the [MIT License](LICENSE).
