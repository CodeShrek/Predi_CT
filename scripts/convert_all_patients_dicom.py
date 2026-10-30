#!/usr/bin/env python3
"""
PrediCT — Convert ALL processed patients to DICOM
===================================================
Discovers every patient that has a _synthetic_coca.nii.gz file across
all known output directories, then converts each one to a proper DICOM
series using the nifti_to_dicom module.

Handles two folder layouts automatically:
  STRUCTURED:  <outputs_dir>/<patient_id>/phase4/<patient_id>_synthetic_coca.nii.gz
  FLAT:        <batch_dir>/<patient_id>_synthetic_coca.nii.gz

Run:
    python scripts/convert_all_patients_dicom.py
"""

import json
import logging
import sys
from pathlib import Path

# ── inject repo root on path so we can import nifti_to_dicom ──────────────
REPO = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from nifti_to_dicom import convert_patient, write_dicom_series, load_nifti, new_uid

import datetime
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("predict.convert_all")

# ── Output directories to scan ────────────────────────────────────────────
SCAN_DIRS = [
    REPO / "outputs",
    REPO / "physio_twin_batch_5",
]


def find_all_patients():
    """
    Return a list of dicts:
      { patient_id, nifti_path, mask_path (or None), out_dir }
    Deduplicates by patient_id — prefers the structured layout.
    """
    seen = {}   # patient_id → entry

    for scan_dir in SCAN_DIRS:
        if not scan_dir.exists():
            continue

        coca_files = sorted(scan_dir.rglob("*_synthetic_coca.nii.gz"))
        for coca in coca_files:
            # Derive patient_id from filename (prefix before first _)
            pid = coca.name.split("_synthetic_coca")[0]
            if not pid:
                continue

            # Look for paired calcium mask
            mask = coca.parent / f"{pid}_synthetic_calcium_mask.nii.gz"
            if not mask.exists():
                mask = None

            # DICOM output goes next to the NIfTI (in a dicom/ sub-folder)
            if coca.parent.name == "phase4":
                # structured layout → dicom goes in patient root
                out_dir = coca.parent.parent / "dicom"
            else:
                # flat layout → dicom goes in a sub-folder next to the file
                out_dir = coca.parent / f"{pid}_dicom"

            entry = {
                "patient_id":  pid,
                "nifti_path":  coca,
                "mask_path":   mask,
                "out_dir":     out_dir,
                "source_dir":  scan_dir.name,
            }

            # Prefer structured layout (outputs/) over flat (batch dirs)
            if pid not in seen or scan_dir.name == "outputs":
                seen[pid] = entry

    return list(seen.values())


def convert_flat_patient(entry: dict) -> Path:
    """
    Lightweight converter for flat-layout patients that don't have a
    summary_metrics.json.  Delegates to write_dicom_series directly.
    """
    pid        = entry["patient_id"]
    nifti_path = entry["nifti_path"]
    mask_path  = entry["mask_path"]
    out_dir    = entry["out_dir"]

    study_uid  = new_uid()
    now        = datetime.datetime.now()
    study_date = now.strftime("%Y%m%d")
    study_time = now.strftime("%H%M%S")

    log.info("=" * 60)
    log.info(f"Patient   : {pid}  [{entry['source_dir']}]")
    log.info(f"Source    : {nifti_path}")
    log.info(f"Output    : {out_dir}")

    # ── Series 1: NCCT ────────────────────────────────────────────────────
    data_zyx, affine, spacing = load_nifti(nifti_path)
    log.info(f"  Shape ZYX: {data_zyx.shape}  Spacing XYZ: {spacing}")

    series1_uid = new_uid()
    series1_dir = out_dir / "synthetic_coca"
    n1 = write_dicom_series(
        data_zyx, affine, spacing, series1_dir,
        patient_id         = pid,
        study_uid          = study_uid,
        series_uid         = series1_uid,
        series_description = "PrediCT Synthetic NCCT — Calcium Augmented",
        series_number      = 1,
        study_date         = study_date,
        study_time         = study_time,
        image_comment      = "PrediCT Synthetic CT — research use only",
    )
    log.info(f"  ✓ NCCT series: {n1} slices → {series1_dir}")

    # ── Series 2: Calcium mask (optional) ─────────────────────────────────
    n2, series2_uid, series2_dir = 0, None, None
    if mask_path and mask_path.exists():
        log.info(f"Loading calcium mask: {mask_path}")
        mask_zyx, mask_affine, mask_spacing = load_nifti(mask_path)
        mask_hu = mask_zyx * 1000.0

        series2_uid = new_uid()
        series2_dir = out_dir / "calcium_mask"
        n2 = write_dicom_series(
            mask_hu, mask_affine, mask_spacing, series2_dir,
            patient_id         = pid,
            study_uid          = study_uid,
            series_uid         = series2_uid,
            series_description = "PrediCT Synthetic Calcium Mask",
            series_number      = 2,
            study_date         = study_date,
            study_time         = study_time,
            image_comment      = "Calcium lesion segmentation mask — research use only",
        )
        log.info(f"  ✓ Mask series: {n2} slices → {series2_dir}")

    # ── Manifest ──────────────────────────────────────────────────────────
    manifest = {
        "patient_id":           pid,
        "conversion_timestamp": now.isoformat(),
        "source_nifti":         str(nifti_path),
        "source_dir":           entry["source_dir"],
        "study_uid":            study_uid,
        "agatston_score":       None,
        "series": [
            {"series_uid": series1_uid, "description": "Synthetic NCCT",
             "directory":  str(series1_dir), "slices": n1},
        ],
    }
    if series2_uid:
        manifest["series"].append({
            "series_uid": series2_uid, "description": "Calcium Mask",
            "directory":  str(series2_dir), "slices": n2,
        })

    manifest_path = out_dir / "dicom_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info(f"  ✓ Manifest: {manifest_path}")

    return out_dir


def main():
    patients = find_all_patients()
    log.info(f"Found {len(patients)} unique patient(s) with synthetic COCA scans.\n")

    results = []
    for entry in patients:
        pid = entry["patient_id"]
        try:
            # Use the structured convert_patient() for the outputs/ folder
            # (it reads summary_metrics.json for the Agatston score)
            if entry["source_dir"] == "outputs":
                out = convert_patient(
                    pid,
                    REPO / "outputs",
                    nifti_path = entry["nifti_path"],
                    mask_path  = entry["mask_path"],
                    out_dir    = entry["out_dir"],
                    also_convert_mask = True,
                )
            else:
                out = convert_flat_patient(entry)

            results.append((pid, entry["source_dir"], "OK", str(out)))
        except Exception as exc:
            log.error(f"FAILED {pid}: {exc}", exc_info=True)
            results.append((pid, entry["source_dir"], "FAILED", str(exc)))

    # ── Final summary ──────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("FINAL CONVERSION SUMMARY")
    log.info("=" * 70)
    ok    = [r for r in results if r[2] == "OK"]
    fails = [r for r in results if r[2] == "FAILED"]

    for pid, src, status, info in results:
        icon = "✓" if status == "OK" else "✗"
        log.info(f"  {icon}  {pid:20s}  [{src:25s}]  {status}")

    log.info("")
    log.info(f"  Total: {len(results)}   ✓ OK: {len(ok)}   ✗ FAILED: {len(fails)}")

    if fails:
        log.info("\nFailed patients:")
        for pid, src, status, msg in fails:
            log.info(f"  {pid} ({src}): {msg}")

    log.info("\nDICOM files are in the dicom/ subfolder inside each patient's output directory.")
    log.info("Open the dicom/ folder in OsiriX, Horos, 3D Slicer, or RadiAnt to view.")


if __name__ == "__main__":
    main()
