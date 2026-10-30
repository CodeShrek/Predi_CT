#!/usr/bin/env python3
"""
PrediCT — NIfTI to DICOM Converter
====================================
Converts synthesized non-contrast CT scans (.nii.gz) from Phase 4 of the
PrediCT pipeline into a valid DICOM series (.dcm files), one file per
axial slice.

Two DICOM series are produced per patient:
  1. Synthetic NCCT scan (calcium-augmented) — loadable in any CT viewer
  2. Calcium mask (optional) — clearly shows the synthetic lesion location

Usage:
    # Convert a single patient by ID (auto-discovers NIfTI files):
    python nifti_to_dicom.py --patient_id 0b49ecbbcf6f

    # Convert ALL patients in the outputs directory:
    python nifti_to_dicom.py --all

    # Specify paths explicitly:
    python nifti_to_dicom.py --nifti /path/to/scan.nii.gz --patient_id MY_ID

    # Skip the calcium mask series:
    python nifti_to_dicom.py --patient_id 0b49ecbbcf6f --no_mask

Dependencies:
    pip install nibabel pydicom numpy
"""

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

import numpy as np

try:
    import nibabel as nib
except ImportError:
    sys.exit("ERROR: nibabel not installed.  Run: pip install nibabel")

try:
    import pydicom
    from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    from pydicom.valuerep import PersonName
except ImportError:
    sys.exit("ERROR: pydicom not installed.  Run: pip install pydicom")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("predict.nifti2dcm")

# UID root (registered to PrediCT research — fine for non-clinical use)
UID_ROOT = "1.2.826.0.1.3680043.10.474"
CT_STORAGE_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.2"


def new_uid() -> str:
    return generate_uid(prefix=UID_ROOT + ".")


# ---------------------------------------------------------------------------
# NIfTI helpers
# ---------------------------------------------------------------------------

def load_nifti(path: Path):
    """
    Load a NIfTI file.

    Returns
    -------
    data_zyx : np.ndarray  shape (Z, Y, X)  float32
    affine   : 4×4 affine matrix (RAS convention, as from nibabel)
    spacing  : (sx, sy, sz) voxel spacings in mm
    """
    img = nib.load(str(path))
    # nibabel stores as (X, Y, Z); we need (Z, Y, X) for axial iteration
    data = np.asarray(img.dataobj, dtype=np.float32)
    data_zyx = np.transpose(data, (2, 1, 0))
    zooms = img.header.get_zooms()
    spacing = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
    return data_zyx, img.affine, spacing


def ras_to_lps(v: np.ndarray) -> np.ndarray:
    """Flip R→L and A→P to convert RAS (NIfTI) → LPS (DICOM)."""
    v = np.array(v, dtype=float)
    v[0] *= -1
    v[1] *= -1
    return v


def orientation_cosines(affine: np.ndarray):
    """Return the 6-value ImageOrientationPatient list in LPS."""
    row_ras = affine[:3, 0] / np.linalg.norm(affine[:3, 0])
    col_ras = affine[:3, 1] / np.linalg.norm(affine[:3, 1])
    row_lps = ras_to_lps(row_ras)
    col_lps = ras_to_lps(col_ras)
    return [round(float(x), 6) for x in (*row_lps, *col_lps)]


# ---------------------------------------------------------------------------
# DICOM writing
# ---------------------------------------------------------------------------

def write_dicom_series(
    data_zyx: np.ndarray,
    affine: np.ndarray,
    spacing,
    out_dir: Path,
    *,
    patient_id: str,
    study_uid: str,
    series_uid: str,
    series_description: str,
    series_number: int,
    study_date: str,
    study_time: str,
    image_comment: str = "",
) -> int:
    """
    Write one .dcm file per axial slice.

    Pixel encoding: 16-bit unsigned, RescaleIntercept = -1024
    (i.e.  stored_value = HU_value + 1024).

    Returns
    -------
    Number of slices written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    iop = orientation_cosines(affine)
    origin_lps = ras_to_lps(affine[:3, 3])
    normal_ras = affine[:3, 2] / np.linalg.norm(affine[:3, 2])
    normal_lps = ras_to_lps(normal_ras)

    n_slices = data_zyx.shape[0]
    HU_INTERCEPT = -1024.0

    log.info(f"    Writing {n_slices} slices → {out_dir}")

    for z in range(n_slices):
        # ── pixel data ──────────────────────────────────────────────────────
        slice_hu = np.clip(data_zyx[z].astype(np.float32), HU_INTERCEPT, HU_INTERCEPT + 65535)
        pixel_array = (slice_hu - HU_INTERCEPT).astype(np.uint16)
        rows, cols = pixel_array.shape

        # ── geometry ────────────────────────────────────────────────────────
        sop_uid = new_uid()
        pos_lps = origin_lps + z * spacing[2] * normal_lps

        # ── file meta ───────────────────────────────────────────────────────
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID    = CT_STORAGE_SOP_CLASS
        file_meta.MediaStorageSOPInstanceUID = sop_uid
        file_meta.TransferSyntaxUID          = ExplicitVRLittleEndian
        file_meta.ImplementationClassUID     = UID_ROOT + ".1"
        file_meta.ImplementationVersionName  = "PREDICT_1.0"

        # ── dataset ─────────────────────────────────────────────────────────
        ds = Dataset()

        # Patient module
        ds.PatientName      = PersonName(f"PrediCT^{patient_id}")
        ds.PatientID        = patient_id
        ds.PatientBirthDate = ""
        ds.PatientSex       = "U"
        ds.PatientComments  = (
            "SYNTHETIC DATA — Generated by PrediCT pipeline. "
            "NOT a real patient. For AI research and training only."
        )

        # General Study module
        ds.StudyInstanceUID        = study_uid
        ds.StudyDate               = study_date
        ds.StudyTime               = study_time
        ds.StudyID                 = "1"
        ds.StudyDescription        = "Synthetic Non-Contrast Cardiac CT (PrediCT)"
        ds.AccessionNumber         = patient_id[:8].upper()
        ds.ReferringPhysicianName  = "PrediCT"

        # General Series module
        ds.Modality             = "CT"
        ds.SeriesInstanceUID    = series_uid
        ds.SeriesNumber         = series_number
        ds.SeriesDescription    = series_description
        ds.BodyPartExamined     = "CHEST"
        ds.PatientPosition      = "HFS"

        # Frame of Reference
        ds.FrameOfReferenceUID        = new_uid()
        ds.PositionReferenceIndicator = ""

        # Image Plane module
        ds.PixelSpacing              = [round(spacing[1], 6), round(spacing[0], 6)]
        ds.SliceThickness            = round(spacing[2], 6)
        ds.ImageOrientationPatient   = iop
        ds.ImagePositionPatient      = [round(float(pos_lps[0]), 4),
                                        round(float(pos_lps[1]), 4),
                                        round(float(pos_lps[2]), 4)]
        ds.SliceLocation             = round(float(pos_lps[2]), 4)

        # CT Image module
        ds.RescaleIntercept          = HU_INTERCEPT
        ds.RescaleSlope              = 1.0
        ds.RescaleType               = "HU"
        ds.WindowCenter              = 400
        ds.WindowWidth               = 1500
        ds.KVP                       = ""
        ds.ImageComments             = image_comment

        # Equipment module
        ds.Manufacturer              = "PrediCT"
        ds.ManufacturerModelName     = "PhysioTwin v1.0"
        ds.SoftwareVersions          = "1.0"
        ds.InstitutionName           = "PrediCT Research"

        # SOP Common
        ds.SOPClassUID               = CT_STORAGE_SOP_CLASS
        ds.SOPInstanceUID            = sop_uid
        ds.InstanceNumber            = z + 1
        ds.SpecificCharacterSet      = "ISO_IR 6"

        # Image Pixel module
        ds.Rows                      = rows
        ds.Columns                   = cols
        ds.BitsAllocated             = 16
        ds.BitsStored                = 16
        ds.HighBit                   = 15
        ds.PixelRepresentation       = 0   # unsigned
        ds.SamplesPerPixel           = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelData                 = pixel_array.tobytes()

        # ── write ───────────────────────────────────────────────────────────
        filename = out_dir / f"slice_{z + 1:04d}.dcm"
        dcm = FileDataset(
            str(filename), ds,
            file_meta=file_meta,
            preamble=b"\x00" * 128,
        )
        # Use modern pydicom v4 API (suppress deprecation warnings)
        dcm.save_as(str(filename), enforce_file_format=True)

        if (z + 1) % 50 == 0 or (z + 1) == n_slices:
            log.info(f"      {z + 1}/{n_slices} slices done...")

    return n_slices


# ---------------------------------------------------------------------------
# Per-patient conversion
# ---------------------------------------------------------------------------

def convert_patient(
    patient_id: str,
    outputs_dir: Path,
    nifti_path: Path | None = None,
    mask_path:  Path | None = None,
    out_dir:    Path | None = None,
    also_convert_mask: bool = True,
) -> Path:
    """
    Convert one patient's NIfTI outputs to DICOM.

    Auto-discovers file paths from the standard outputs/<patient_id>/ layout
    when nifti_path / mask_path are not given explicitly.
    """
    patient_dir = outputs_dir / patient_id

    # ── auto-discover NIfTI ──────────────────────────────────────────────
    if nifti_path is None:
        for candidate in [
            patient_dir / "phase4" / f"{patient_id}_synthetic_coca.nii.gz",
            patient_dir / f"{patient_id}_synthetic_coca.nii.gz",
        ]:
            if candidate.exists():
                nifti_path = candidate
                break
        if nifti_path is None:
            raise FileNotFoundError(
                f"Cannot find synthetic COCA NIfTI for patient '{patient_id}' "
                f"under {patient_dir}.  Pass --nifti explicitly."
            )

    # ── auto-discover calcium mask ───────────────────────────────────────
    if mask_path is None and also_convert_mask:
        for candidate in [
            patient_dir / "phase3" / f"{patient_id}_synthetic_calcium_mask.nii.gz",
            patient_dir / f"{patient_id}_synthetic_calcium_mask.nii.gz",
        ]:
            if candidate.exists():
                mask_path = candidate
                break

    if out_dir is None:
        out_dir = patient_dir / "dicom"
    out_dir = Path(out_dir)

    # ── read Agatston score ──────────────────────────────────────────────
    agatston_score = None
    metrics_file = patient_dir / "summary_metrics.json"
    if metrics_file.exists():
        try:
            metrics = json.loads(metrics_file.read_text())
            agatston_score = metrics.get("phase4", {}).get("synthetic_agatston")
        except Exception as exc:
            log.warning(f"Could not read summary_metrics.json: {exc}")

    # ── shared DICOM study info ──────────────────────────────────────────
    study_uid  = new_uid()
    now        = datetime.datetime.now()
    study_date = now.strftime("%Y%m%d")
    study_time = now.strftime("%H%M%S")

    log.info("=" * 60)
    log.info(f"Patient   : {patient_id}")
    log.info(f"Source    : {nifti_path}")
    log.info(f"Output    : {out_dir}")
    if agatston_score is not None:
        log.info(f"Agatston  : {agatston_score:.1f}")

    # ── Series 1 — Synthetic NCCT ────────────────────────────────────────
    log.info("Loading synthetic COCA NIfTI...")
    data_zyx, affine, spacing = load_nifti(nifti_path)
    log.info(
        f"  Shape (Z,Y,X): {data_zyx.shape}  "
        f"Spacing (X,Y,Z): {spacing[0]:.3f} × {spacing[1]:.3f} × {spacing[2]:.3f} mm"
    )

    series1_uid = new_uid()
    series1_dir = out_dir / "synthetic_coca"
    comment1    = (
        f"Synthetic Agatston Score: {agatston_score:.1f}"
        if agatston_score is not None
        else "PrediCT Synthetic NCCT — research use only"
    )
    n1 = write_dicom_series(
        data_zyx, affine, spacing, series1_dir,
        patient_id       = patient_id,
        study_uid        = study_uid,
        series_uid       = series1_uid,
        series_description = "PrediCT Synthetic NCCT — Calcium Augmented",
        series_number    = 1,
        study_date       = study_date,
        study_time       = study_time,
        image_comment    = comment1,
    )
    log.info(f"  ✓ NCCT series: {n1} slices → {series1_dir}")

    # ── Series 2 — Calcium mask (optional) ───────────────────────────────
    n2, series2_uid, series2_dir = 0, None, None
    if also_convert_mask and mask_path and Path(mask_path).exists():
        log.info(f"Loading calcium mask: {mask_path}")
        mask_zyx, mask_affine, mask_spacing = load_nifti(mask_path)
        # Map 0→1 mask values to HU: 0 = background (−1024 stored), 1 = 1000 HU
        mask_hu = mask_zyx * 1000.0

        series2_uid = new_uid()
        series2_dir = out_dir / "calcium_mask"
        n2 = write_dicom_series(
            mask_hu, mask_affine, mask_spacing, series2_dir,
            patient_id         = patient_id,
            study_uid          = study_uid,
            series_uid         = series2_uid,
            series_description = "PrediCT Synthetic Calcium Mask",
            series_number      = 2,
            study_date         = study_date,
            study_time         = study_time,
            image_comment      = "Calcium lesion segmentation mask — research use only",
        )
        log.info(f"  ✓ Mask series: {n2} slices → {series2_dir}")

    # ── manifest ─────────────────────────────────────────────────────────
    manifest = {
        "patient_id":             patient_id,
        "conversion_timestamp":   now.isoformat(),
        "source_nifti":           str(nifti_path),
        "study_uid":              study_uid,
        "agatston_score":         agatston_score,
        "series": [
            {
                "series_uid":   series1_uid,
                "description":  "Synthetic NCCT",
                "directory":    str(series1_dir),
                "slices":       n1,
            }
        ],
    }
    if series2_uid:
        manifest["series"].append({
            "series_uid":  series2_uid,
            "description": "Calcium Mask",
            "directory":   str(series2_dir),
            "slices":      n2,
        })

    manifest_path = out_dir / "dicom_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info(f"  ✓ Manifest: {manifest_path}")

    return out_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PrediCT NIfTI → DICOM Converter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python nifti_to_dicom.py --all
  python nifti_to_dicom.py --patient_id 0b49ecbbcf6f
  python nifti_to_dicom.py --nifti /path/scan.nii.gz --patient_id MY_ID
  python nifti_to_dicom.py --patient_id 0b49ecbbcf6f --no_mask
""",
    )
    parser.add_argument("--patient_id",  default=None,
                        help="Patient ID (sub-folder under outputs_dir)")
    parser.add_argument("--outputs_dir", default=str(Path(__file__).parent.parent / "outputs"),
                        help="Root outputs directory (default: ../outputs/)")
    parser.add_argument("--nifti",       default=None,
                        help="Explicit NIfTI path (skips auto-discovery)")
    parser.add_argument("--mask",        default=None,
                        help="Explicit calcium mask NIfTI path (optional)")
    parser.add_argument("--out_dir",     default=None,
                        help="DICOM output directory (default: outputs/<patient_id>/dicom/)")
    parser.add_argument("--no_mask",     action="store_true",
                        help="Skip the calcium mask DICOM series")
    parser.add_argument("--all",         action="store_true",
                        help="Convert ALL patient folders in outputs_dir")

    args = parser.parse_args()
    outputs_dir = Path(args.outputs_dir)

    if not outputs_dir.exists():
        sys.exit(f"ERROR: outputs_dir not found: {outputs_dir}")

    if args.all:
        patient_dirs = sorted(d for d in outputs_dir.iterdir() if d.is_dir())
        if not patient_dirs:
            sys.exit(f"No patient directories found in {outputs_dir}")

        log.info(f"Batch converting {len(patient_dirs)} patient(s)...")
        results = []
        for pdir in patient_dirs:
            try:
                out = convert_patient(
                    pdir.name, outputs_dir,
                    also_convert_mask=not args.no_mask,
                )
                results.append((pdir.name, "OK", str(out)))
            except Exception as exc:
                log.error(f"FAILED {pdir.name}: {exc}")
                results.append((pdir.name, "FAILED", str(exc)))

        log.info("\n" + "=" * 60)
        log.info("BATCH SUMMARY")
        log.info("=" * 60)
        for pid, status, info in results:
            log.info(f"  {pid:20s}  {status:6s}  {info}")
    else:
        if args.patient_id is None and args.nifti is None:
            parser.error("Provide --patient_id, --nifti, or --all")

        pid = args.patient_id
        if pid is None:
            # derive from filename:  0b49ecbbcf6f_synthetic_coca.nii.gz → 0b49ecbbcf6f
            pid = Path(args.nifti).name.split("_")[0]
            log.info(f"Auto-derived patient_id from filename: {pid}")

        convert_patient(
            pid, outputs_dir,
            nifti_path = Path(args.nifti)  if args.nifti   else None,
            mask_path  = Path(args.mask)   if args.mask    else None,
            out_dir    = Path(args.out_dir) if args.out_dir else None,
            also_convert_mask = not args.no_mask,
        )

    log.info("Done.")


if __name__ == "__main__":
    main()
