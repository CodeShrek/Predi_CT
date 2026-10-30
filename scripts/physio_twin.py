"""
Physio-Twin: Unified Generative Digital Twin for Coronary Calcium Synthesis
=============================================================================
End-to-end multi-physics pipeline connecting:
  - Phase 1: Anatomical Scaffolding (Deformable B-Spline + Maurer SDT + Curvature Flow)
  - Phase 2: Hemodynamic Surrogate (PINN Navier-Stokes solving Endothelial Shear Stress)
  - Phase 3: Stochastic Plaque Growth (SDE biased by low ESS)
  - Phase 4: Physiological Non-Contrast CT Texturing (Continuous HU Alpha-Blending)

All outputs are neatly organized patient-wise:
  outputs/{patient_id}/
    ├── phase1/ (vessel scaffold, CT, ground truth calcium, warped atlas)
    ├── phase2/ (ESS CSV, ESS field NIfTI, Paraview VTP)
    ├── phase3/ (synthetic calcium mask)
    ├── phase4/ (final synthetic COCA CT scan)
    ├── summary_metrics.json
    └── Top-level convenience copies of key deliverables
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.interpolate import griddata

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("PhysioTwin")

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "phase1_segmentation"))


def interpolate_ess_to_grid(csv_path: Path, mask_path: Path, out_path: Path) -> Path:
    """Interpolates sparse point cloud ESS values onto a dense 3D voxel grid."""
    logger.info(f"Interpolating ESS point cloud {csv_path.name} to dense grid...")
    df = pd.read_csv(csv_path)
    points = df[['x', 'y', 'z']].values
    values = df['ess_magnitude'].values

    mask_img = sitk.ReadImage(str(mask_path))
    mask_arr = sitk.GetArrayFromImage(mask_img)
    spacing = mask_img.GetSpacing()
    origin = mask_img.GetOrigin()

    z_dim, y_dim, x_dim = mask_arr.shape
    z, y, x = np.mgrid[0:z_dim, 0:y_dim, 0:x_dim]

    grid_coords = np.column_stack([
        x.ravel() * spacing[0] + origin[0],
        y.ravel() * spacing[1] + origin[1],
        z.ravel() * spacing[2] + origin[2]
    ])

    mask_flat = mask_arr.ravel()
    valid_idx = np.where(mask_flat > 0)[0]

    if len(valid_idx) == 0:
        logger.warning("Vessel mask is empty. Writing zero ESS field.")
        ess_arr = np.zeros_like(mask_arr, dtype=float)
    else:
        valid_coords = grid_coords[valid_idx]
        interp_vals = griddata(points, values, valid_coords, method='nearest')
        ess_arr = np.zeros_like(mask_arr, dtype=float)
        ess_arr_flat = ess_arr.ravel()
        ess_arr_flat[valid_idx] = interp_vals
        ess_arr = ess_arr_flat.reshape(mask_arr.shape)

    ess_img = sitk.GetImageFromArray(ess_arr)
    ess_img.CopyInformation(mask_img)
    sitk.WriteImage(ess_img, str(out_path))
    logger.info(f"Saved dense ESS field to {out_path}")
    return out_path


def run_physio_twin(
    coca_patient_dir: Path,
    target_agatston: int = 400,
    out_dir: Path = Path("outputs"),
    epochs: int = None,
    phase1_only: bool = False,
    skip_phase1_if_exists: bool = True
) -> dict:
    """Executes full Phases 1-4 generative pipeline for a target COCA patient."""
    t0_total = time.time()
    coca_patient_dir = coca_patient_dir.resolve()
    patient_id = coca_patient_dir.name
    logger.info(f"====================================================================")
    logger.info(f"   PHYSIO-TWIN PIPELINE: Patient {patient_id}")
    logger.info(f"====================================================================")

    # 1. Setup Patient-Specific Directory Structure
    patient_out_dir = (out_dir / patient_id).resolve()
    phase1_dir = patient_out_dir / "phase1"
    phase2_dir = patient_out_dir / "phase2"
    phase3_dir = patient_out_dir / "phase3"
    phase4_dir = patient_out_dir / "phase4"

    for d in [phase1_dir, phase2_dir, phase3_dir, phase4_dir]:
        d.mkdir(parents=True, exist_ok=True)

    metrics = {
        "patient_id": patient_id,
        "target_agatston": target_agatston,
        "phase1": {},
        "phase2": {},
        "phase3": {},
        "phase4": {}
    }

    # =========================================================
    # Phase 1: Anatomical Scaffolding
    # =========================================================
    logger.info("--- [Phase 1] Anatomical Scaffolding ---")
    vessel_mask_path = phase1_dir / f"{patient_id}_synthetic_vessel.nii.gz"
    patient_ct_path = phase1_dir / f"{patient_id}_patient_ct.nii.gz"
    gt_calcium_path = phase1_dir / f"{patient_id}_ground_truth_calcium.nii.gz"

    # Also check if existing in patient root or output_bspline_v3
    if not vessel_mask_path.exists():
        fallback_root = patient_out_dir / f"{patient_id}_synthetic_vessel.nii.gz"
        bspline_fallback = REPO_ROOT / "output_bspline_v3" / f"{patient_id}_synthetic_vessel.nii.gz"
        if fallback_root.exists():
            shutil.copy2(fallback_root, vessel_mask_path)
        elif bspline_fallback.exists():
            shutil.copy2(bspline_fallback, vessel_mask_path)
            for f in (REPO_ROOT / "output_bspline_v3").glob(f"{patient_id}_*"):
                shutil.copy2(f, phase1_dir / f.name)

    run_phase1 = True
    if vessel_mask_path.exists() and skip_phase1_if_exists:
        # Check volume validity
        chk_img = sitk.ReadImage(str(vessel_mask_path))
        chk_vox = int(sitk.GetArrayFromImage(chk_img).sum())
        if chk_vox >= 1000:
            logger.info(f"Reusing verified Phase 1 vessel scaffold: {vessel_mask_path} ({chk_vox} voxels)")
            metrics["phase1"]["vessel_voxels"] = chk_vox
            metrics["phase1"]["status"] = "reused_verified"
            run_phase1 = False
        else:
            logger.warning(f"Existing vessel mask has only {chk_vox} voxels (< 1000). Re-running Phase 1...")

    if run_phase1:
        logger.info("Executing Phase 1 B-Spline Registration + Level-Set Warping...")
        from masking import (
            Preprocessor, RegistrationEngine, LabelTransformer,
            get_cardiac_roi, select_atlases_by_ncc, discover_atlas_cases, CFG,
            PlaqueValidator
        )
        ncct_path = coca_patient_dir / f"{patient_id}_raw_img.nii.gz"
        if not ncct_path.exists():
            ncct_path = coca_patient_dir / f"{patient_id}_img.nii.gz"
        coca_seg = coca_patient_dir / f"{patient_id}_seg.nii.gz"

        patient_pack = Preprocessor.run_memory_resample(ncct_path, coca_seg)
        fixed_win = patient_pack["reg_win"]
        eval_fixed_raw = patient_pack["eval_raw"]
        eval_coca_seg = patient_pack["eval_lbl"]

        sitk.WriteImage(eval_fixed_raw, str(patient_ct_path))
        sitk.WriteImage(eval_coca_seg, str(gt_calcium_path))

        atlas_pool = discover_atlas_cases(CFG.IMAGECAS_DIR)
        atlas_reg_wins, atlas_lbl_evals, atlas_lbl_regs = {}, {}, {}
        for case in atlas_pool:
            cid = case["id"]
            pack = Preprocessor.run_memory_resample(case["img"], case["lbl"])
            atlas_reg_wins[cid] = pack["reg_win"]
            atlas_lbl_evals[cid] = pack["eval_lbl"]
            atlas_lbl_regs[cid] = Preprocessor.resample_volume(
                Preprocessor.orient_to_lps(sitk.ReadImage(str(case["lbl"]), sitk.sitkUInt8)),
                CFG.REG_SPACING_MM, is_label=True
            )

        selected_atlases = select_atlases_by_ncc(fixed_win, atlas_pool, atlas_reg_wins, CFG.N_ATLASES)
        transformed_labels = []
        per_atlas_results = []
        per_atlas_transforms = {}

        for aidx, atlas_case in enumerate(selected_atlases, 1):
            cid = atlas_case["id"]
            atlas_win_reg = atlas_reg_wins[cid]
            atlas_lbl_reg = atlas_lbl_regs[cid]
            atlas_lbl_eval = atlas_lbl_evals[cid]

            rigid_tx, rigid_mi = RegistrationEngine.run_rigid_stage(fixed_win, atlas_win_reg)
            rigid_atlas_lbl = sitk.Resample(atlas_lbl_reg, fixed_win, rigid_tx, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
            rigid_lbl_arr = sitk.GetArrayFromImage(rigid_atlas_lbl)
            if np.count_nonzero(rigid_lbl_arr) == 0:
                per_atlas_results.append({"atlas_id": cid, "final_mi": None, "n_warp": 0})
                continue

            roi_fixed = get_cardiac_roi(rigid_lbl_arr, CFG.ROI_MARGIN_MM, CFG.REG_SPACING_MM)
            if roi_fixed is None:
                per_atlas_results.append({"atlas_id": cid, "final_mi": None, "n_warp": 0})
                continue

            final_tx, affine_mi = RegistrationEngine.run_affine_stage(fixed_win, atlas_win_reg, rigid_tx, roi_fixed, rigid_mi=rigid_mi)
            if CFG.BSPLINE_ENABLED:
                final_tx, bspline_mi = RegistrationEngine.run_bspline_stage(fixed_win, atlas_win_reg, final_tx, roi_fixed, affine_mi=affine_mi)
                final_mi = bspline_mi
            else:
                final_mi = affine_mi

            warped_lbl = LabelTransformer.transform_labels(atlas_lbl_eval, eval_fixed_raw, final_tx)
            warp_arr = sitk.GetArrayFromImage(warped_lbl)
            n_warp = int(np.count_nonzero(warp_arr))
            if n_warp > 0:
                transformed_labels.append(warped_lbl)
            per_atlas_results.append({"atlas_id": cid, "final_mi": final_mi, "n_warp": n_warp})
            per_atlas_transforms[cid] = final_tx

        valid_results = [r for r in per_atlas_results if r.get("final_mi") is not None and not np.isnan(r["final_mi"]) and r["n_warp"] > 0]
        if not valid_results:
            raise RuntimeError(f"Phase 1 failed: No atlas achieved valid registration for patient {patient_id}.")

        best = min(valid_results, key=lambda r: r["final_mi"])
        best_label_idx = [r["atlas_id"] for r in valid_results].index(best["atlas_id"])
        best_vessels = transformed_labels[best_label_idx]

        # Topology smoothing (Curvature Flow + Maurer SDT)
        cc_filter = sitk.ConnectedComponentImageFilter()
        labeled_img = cc_filter.Execute(best_vessels)
        relabel_filter = sitk.RelabelComponentImageFilter()
        relabel_filter.SetMinimumObjectSize(15)
        cleaned = relabel_filter.Execute(labeled_img)
        cleaned = sitk.BinaryThreshold(cleaned, lowerThreshold=1, upperThreshold=10000, insideValue=1, outsideValue=0)
        cleaned = sitk.Cast(cleaned, sitk.sitkUInt8)

        float_img = sitk.Cast(cleaned, sitk.sitkFloat32)
        smoothed_float = sitk.CurvatureFlow(float_img, timeStep=0.0625, numberOfIterations=2)
        smoothed = sitk.BinaryThreshold(smoothed_float, lowerThreshold=0.35, insideValue=1, outsideValue=0)
        smoothed = sitk.Cast(smoothed, sitk.sitkUInt8)
        smoothed.CopyInformation(best_vessels)
        best_vessels = smoothed

        sitk.WriteImage(best_vessels, str(vessel_mask_path))
        vox_cnt = int(sitk.GetArrayFromImage(best_vessels).sum())
        logger.info(f"Phase 1 completed. Saved vessel scaffold ({vox_cnt} voxels) to {vessel_mask_path}")
        metrics["phase1"] = {"status": "computed", "best_atlas": best["atlas_id"], "vessel_voxels": vox_cnt}

    # Ensure patient top-level has convenience copy
    shutil.copy2(vessel_mask_path, patient_out_dir / f"{patient_id}_synthetic_vessel.nii.gz")

    if phase1_only:
        logger.info(f"Stopping after Phase 1 as requested (--phase1_only). Patient {patient_id} complete.")
        return metrics

    # =========================================================
    # Phase 2: Hemodynamics PINN
    # =========================================================
    logger.info("--- [Phase 2] Hemodynamic Navier-Stokes PINN ---")
    pinn_csv = phase2_dir / f"{patient_id}_ess_predictions.csv"
    pinn_vtp = phase2_dir / f"{patient_id}_ess_predictions.vtp"
    ess_nifti_path = phase2_dir / f"{patient_id}_ess_field.nii.gz"

    env = os.environ.copy()
    env["PREDICT_MASK_PATH"] = str(vessel_mask_path.resolve())
    env["PREDICT_EXPORT_DIR"] = str(phase2_dir.resolve())
    env["PREDICT_CHECKPOINT_DIR"] = str((phase2_dir / "checkpoints").resolve())
    env["PREDICT_LOG_DIR"] = str((phase2_dir / "logs").resolve())
    if epochs is not None:
        env["PREDICT_EPOCHS"] = str(epochs)

    p2_t0 = time.time()
    logger.info(f"Running PINN solver (epochs={env.get('PREDICT_EPOCHS', 'auto')})...")
    subprocess.run(
        [sys.executable, "run_phase2.py"],
        env=env,
        cwd=str(REPO_ROOT / "src" / "phase2_hemodynamics"),
        check=True
    )

    # run_phase2 exports to export_dir / ess_predictions.csv
    raw_csv = phase2_dir / "ess_predictions.csv"
    raw_vtp = phase2_dir / "ess_predictions.vtp"
    if raw_csv.exists() and not pinn_csv.exists():
        raw_csv.rename(pinn_csv)
    if raw_vtp.exists() and not pinn_vtp.exists():
        raw_vtp.rename(pinn_vtp)

    # Convert sparse point cloud to dense 3D NIfTI
    interpolate_ess_to_grid(pinn_csv, vessel_mask_path, ess_nifti_path)
    shutil.copy2(ess_nifti_path, patient_out_dir / f"{patient_id}_ess_field.nii.gz")

    # Extract Phase 2 metrics
    df_ess = pd.read_csv(pinn_csv)
    mean_ess = float(df_ess['ess_magnitude'].mean())
    atherogenic_pct = float((df_ess['ess_magnitude'] < 1.0).mean() * 100)
    logger.info(f"Phase 2 Validation: Mean ESS = {mean_ess:.4f} Pa (Atherogenic < 1.0 Pa = {atherogenic_pct:.1f}%)")
    metrics["phase2"] = {
        "mean_ess_pa": mean_ess,
        "atherogenic_pct": atherogenic_pct,
        "wall_points": len(df_ess),
        "duration_sec": time.time() - p2_t0
    }

    # =========================================================
    # Phase 3: Stochastic Plaque Growth (SDE)
    # =========================================================
    logger.info("--- [Phase 3] Stochastic Plaque Growth (SDE) ---")
    synth_calcium_path = phase3_dir / f"{patient_id}_synthetic_calcium_mask.nii.gz"

    cmd_p3 = [
        sys.executable, "run_phase3_sde.py",
        "--vessel_mask", str(vessel_mask_path.resolve()),
        "--ess_field", str(ess_nifti_path.resolve()),
        "--output", str(synth_calcium_path.resolve()),
        "--agatston", str(target_agatston)
    ]
    p3_t0 = time.time()
    subprocess.run(cmd_p3, cwd=str(REPO_ROOT / "src" / "phase3_plaque_growth"), check=True)
    shutil.copy2(synth_calcium_path, patient_out_dir / f"{patient_id}_synthetic_calcium_mask.nii.gz")

    ca_img = sitk.ReadImage(str(synth_calcium_path))
    ca_voxels = int((sitk.GetArrayFromImage(ca_img) > 0).sum())
    logger.info(f"Phase 3 completed. Generated {ca_voxels} calcification voxels.")
    metrics["phase3"] = {
        "calcium_voxels": ca_voxels,
        "duration_sec": time.time() - p3_t0
    }

    # =========================================================
    # Phase 4: Physiological Texturing
    # =========================================================
    logger.info("--- [Phase 4] Physiological Texturing ---")
    raw_ncct = coca_patient_dir / f"{patient_id}_raw_img.nii.gz"
    if not raw_ncct.exists():
        raw_ncct = coca_patient_dir / f"{patient_id}_img.nii.gz"

    final_coca_path = phase4_dir / f"{patient_id}_synthetic_coca.nii.gz"
    cmd_p4 = [
        sys.executable, "run_phase4_texture.py",
        "--ncct", str(raw_ncct.resolve()),
        "--synth_mask", str(synth_calcium_path.resolve()),
        "--vessel_mask", str(vessel_mask_path.resolve()),
        "--output", str(final_coca_path.resolve())
    ]
    p4_t0 = time.time()
    res_p4 = subprocess.run(
        cmd_p4,
        cwd=str(REPO_ROOT / "src" / "phase4_texturing"),
        capture_output=True,
        text=True,
        check=True
    )
    shutil.copy2(final_coca_path, patient_out_dir / f"{patient_id}_synthetic_coca.nii.gz")

    # Parse computed synthetic Agatston score from stdout/stderr
    synth_agatston = None
    for line in (res_p4.stderr + res_p4.stdout).splitlines():
        if "Computed Synthetic Agatston Score:" in line:
            synth_agatston = float(line.split(":")[-1].strip())
            break

    logger.info(f"Phase 4 completed! Final Synthetic Agatston Score = {synth_agatston}")
    logger.info(f"Final synthetic CT saved to: {final_coca_path}")
    metrics["phase4"] = {
        "synthetic_agatston": synth_agatston,
        "final_scan": str(final_coca_path),
        "duration_sec": time.time() - p4_t0
    }

    metrics["total_duration_sec"] = time.time() - t0_total

    # Save summary metrics
    summary_path = patient_out_dir / "summary_metrics.json"
    with open(summary_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"====================================================================")
    logger.info(f"   PATIENT {patient_id} PIPELINE COMPLETE! Total time: {metrics['total_duration_sec']:.1f}s")
    logger.info(f"====================================================================")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Physio-Twin: Multi-Physics Coronary Calcium Synthesis Pipeline")
    parser.add_argument("--patient_dir", type=str, default=None, help="Path to target COCA patient directory")
    parser.add_argument("--patients", nargs="*", default=None, help="List of COCA patient IDs or paths to process")
    parser.add_argument("--target_agatston", type=int, default=400, help="Target Agatston score multiplier")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Main outputs directory")
    parser.add_argument("--epochs", type=int, default=None, help="PINN training epochs for Phase 2")
    parser.add_argument("--phase1_only", action="store_true", help="Stop after generating Phase 1 scaffold")
    parser.add_argument("--recompute_phase1", action="store_true", help="Force recomputation of Phase 1 even if scaffold exists")

    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    target_dirs = []
    if args.patients:
        for p in args.patients:
            p_path = Path(p)
            if not p_path.exists():
                p_path = REPO_ROOT / "COCA" / p
            if p_path.exists():
                target_dirs.append(p_path)
            else:
                logger.error(f"Patient directory {p} not found.")
    elif args.patient_dir:
        p_path = Path(args.patient_dir)
        if not p_path.exists():
            p_path = REPO_ROOT / "COCA" / args.patient_dir
        target_dirs.append(p_path)
    else:
        # Default to the 4 target patients
        default_cohort = ["1cc17f65f909", "0b49ecbbcf6f", "0cc7aae764f7", "0f0baa4e72c8"]
        logger.info(f"No patient specified. Defaulting to verified cohort: {default_cohort}")
        for pid in default_cohort:
            p_path = REPO_ROOT / "COCA" / pid
            if p_path.exists():
                target_dirs.append(p_path)

    for p_dir in target_dirs:
        try:
            run_physio_twin(
                coca_patient_dir=p_dir,
                target_agatston=args.target_agatston,
                out_dir=out_dir,
                epochs=args.epochs,
                phase1_only=args.phase1_only,
                skip_phase1_if_exists=not args.recompute_phase1
            )
        except Exception as e:
            logger.error(f"Failed processing patient {p_dir.name}: {e}", exc_info=True)


if __name__ == "__main__":
    main()
