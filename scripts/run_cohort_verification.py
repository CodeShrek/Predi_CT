import sys
sys.path.insert(0, "src/phase1_segmentation")
from pathlib import Path
import time
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from masking_bspline import Preprocessor, LabelTransformer, get_cardiac_roi, CFG, discover_atlas_cases, select_atlases_by_ncc

sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(8)

out_dir = Path("output_bspline_v3")
out_dir.mkdir(parents=True, exist_ok=True)

# Discover atlases
atlas_pool = discover_atlas_cases(CFG.IMAGECAS_DIR)
print(f"Atlas pool: {len(atlas_pool)} available ImageCAS cases.", flush=True)

# Select 3 target patients:
target_patients = ["0b49ecbbcf6f", "0cc7aae764f7", "0f0baa4e72c8"]
print(f"Target patients for verification: {target_patients}\n", flush=True)

cohort_summary = []

for p_idx, pid in enumerate(target_patients, 1):
    print(f"\n{'='*75}", flush=True)
    print(f"[{p_idx}/{len(target_patients)}] Processing Patient: {pid}", flush=True)
    print(f"{'='*75}", flush=True)
    t_start = time.time()

    # 1. Load and resample patient
    img_p = Path(f"COCA/{pid}/{pid}_raw_img.nii.gz")
    seg_p = Path(f"COCA/{pid}/{pid}_seg.nii.gz")

    patient_pack = Preprocessor.run_memory_resample(img_p, seg_p)
    fixed_win = patient_pack["reg_win"]
    eval_fixed_raw = patient_pack["eval_raw"]
    eval_coca_seg = patient_pack["eval_lbl"]

    # Save patient CT & GT calcium
    sitk.WriteImage(eval_fixed_raw, str(out_dir / f"{pid}_patient_ct.nii.gz"))
    sitk.WriteImage(eval_coca_seg, str(out_dir / f"{pid}_ground_truth_calcium.nii.gz"))
    print(f"  [1/5] Loaded patient data and saved CT/calcium volumes.", flush=True)

    # 2. NCC atlas selection (test first 20 cases for fast ranking)
    test_pool = atlas_pool[:20]
    atlas_wins = {}
    for c in test_pool:
        p = Preprocessor.run_memory_resample(c["img"])
        atlas_wins[c["id"]] = p["reg_win"]

    top_cases = select_atlases_by_ncc(fixed_win, test_pool, atlas_wins, n_select=1)
    best_atlas_case = top_cases[0]
    cid = best_atlas_case["id"]
    print(f"  [2/5] Selected Top NCC Atlas: Case {cid}", flush=True)

    # Load atlas label
    atlas_pack = Preprocessor.run_memory_resample(best_atlas_case["img"], best_atlas_case["lbl"])
    atlas_win = atlas_pack["reg_win"]
    atlas_lbl_eval = atlas_pack["eval_lbl"]
    atlas_lbl_reg = Preprocessor.resample_volume(atlas_lbl_eval, CFG.REG_SPACING_MM, is_label=True)

    # 3. Stage 1: Rigid Alignment
    print(f"  [3/5] Running Rigid & Affine Alignment...", flush=True)
    init_tx = sitk.CenteredTransformInitializer(
        fixed_win, atlas_win,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    R_rigid = sitk.ImageRegistrationMethod()
    R_rigid.SetMetricAsMattesMutualInformation(50)
    R_rigid.SetMetricSamplingStrategy(sitk.ImageRegistrationMethod.REGULAR)
    R_rigid.SetMetricSamplingPercentage(0.15)
    R_rigid.SetShrinkFactorsPerLevel([8, 4, 2, 1])
    R_rigid.SetSmoothingSigmasPerLevel([3, 2, 1, 0])
    R_rigid.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    R_rigid.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0, minStep=0.005, numberOfIterations=80, relaxationFactor=0.5
    )
    R_rigid.SetOptimizerScalesFromPhysicalShift()
    R_rigid.SetInitialTransform(init_tx, inPlace=False)
    rigid_tx = R_rigid.Execute(fixed_win, atlas_win)

    # Cardiac ROI & Lung-Heart boundary mask
    rigid_lbl = sitk.Resample(atlas_lbl_reg, fixed_win, rigid_tx, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    rigid_arr = sitk.GetArrayFromImage(rigid_lbl)
    roi = get_cardiac_roi(rigid_arr, margin_mm=25.0, spacing_mm=CFG.REG_SPACING_MM)
    fz0, fz1, fy0, fy1, fx0, fx1 = roi

    fixed_arr = sitk.GetArrayFromImage(fixed_win)
    mask_arr = np.zeros_like(fixed_arr, dtype=np.uint8)
    mask_arr[fz0:fz1, fy0:fy1, fx0:fx1] = 1
    metric_mask = sitk.GetImageFromArray(mask_arr)
    metric_mask.CopyInformation(fixed_win)

    # Stage 2: Affine
    rigid_atlas_win = sitk.Resample(atlas_win, fixed_win, rigid_tx, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    fixed_center = fixed_win.TransformContinuousIndexToPhysicalPoint(
        [(fx0+fx1)/2.0, (fy0+fy1)/2.0, (fz0+fz1)/2.0]
    )
    atx = sitk.AffineTransform(3)
    atx.SetCenter(fixed_center)

    R_affine = sitk.ImageRegistrationMethod()
    R_affine.SetMetricAsMattesMutualInformation(50)
    R_affine.SetMetricSamplingStrategy(sitk.ImageRegistrationMethod.RANDOM)
    R_affine.SetMetricSamplingPercentage(0.15)
    R_affine.SetShrinkFactorsPerLevel([4, 2, 1])
    R_affine.SetSmoothingSigmasPerLevel([2, 1, 0])
    R_affine.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    R_affine.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.5, minStep=0.002, numberOfIterations=80, relaxationFactor=0.7
    )
    R_affine.SetOptimizerScalesFromPhysicalShift()
    R_affine.SetMetricFixedMask(metric_mask)
    R_affine.SetInitialTransform(atx, inPlace=False)
    final_affine = R_affine.Execute(fixed_win, rigid_atlas_win)

    affine_composite_tx = sitk.CompositeTransform(3)
    affine_composite_tx.AddTransform(rigid_tx)
    affine_composite_tx.AddTransform(final_affine)

    # 4. Stage 3: Multi-Resolution Diffeomorphic B-Spline
    print(f"  [4/5] Running Multi-Resolution B-Spline (Coarse-to-Fine)...", flush=True)
    affine_atlas_win = sitk.Resample(atlas_win, fixed_win, affine_composite_tx, sitk.sitkLinear, 0.0, sitk.sitkFloat32)

    base_mesh = [6, 6, 6]
    bspline_tx = sitk.BSplineTransformInitializer(fixed_win, base_mesh, order=3)

    R_bspline = sitk.ImageRegistrationMethod()
    R_bspline.SetMetricAsMattesMutualInformation(50)
    R_bspline.SetMetricSamplingStrategy(sitk.ImageRegistrationMethod.RANDOM)
    R_bspline.SetMetricSamplingPercentage(0.12)
    R_bspline.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.2, minStep=0.002, numberOfIterations=50, relaxationFactor=0.75
    )
    R_bspline.SetOptimizerScalesFromPhysicalShift()
    R_bspline.SetMetricFixedMask(metric_mask)
    R_bspline.SetInitialTransformAsBSpline(bspline_tx, inPlace=True, scaleFactors=[1, 2])
    R_bspline.SetShrinkFactorsPerLevel([2, 1])
    R_bspline.SetSmoothingSigmasPerLevel([1, 0])
    R_bspline.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    R_bspline.SetInterpolator(sitk.sitkLinear)

    final_bspline = R_bspline.Execute(fixed_win, affine_atlas_win)

    # Displacement & Jacobian
    disp_filter = sitk.TransformToDisplacementFieldFilter()
    disp_filter.SetReferenceImage(fixed_win)
    disp_field = disp_filter.Execute(final_bspline)
    disp_arr = sitk.GetArrayFromImage(disp_field)
    max_disp = float(np.sqrt((disp_arr**2).sum(axis=-1)).max())

    jac_filter = sitk.DisplacementFieldJacobianDeterminantFilter()
    jac_img = jac_filter.Execute(disp_field)
    min_jac = float(sitk.GetArrayFromImage(jac_img).min())

    full_composite = sitk.CompositeTransform(3)
    full_composite.AddTransform(affine_composite_tx)
    full_composite.AddTransform(final_bspline)

    # 5. Output Warping, Smoothing & Epicardial Tree Filtering
    print(f"  [5/5] Warping, Filtering Non-Coronary Artifacts & Saving...", flush=True)
    # Warped atlas CT
    warped_atlas_ct = sitk.Resample(atlas_pack["eval_raw"], eval_fixed_raw, full_composite, sitk.sitkLinear, -1024.0, sitk.sitkFloat32)
    sitk.WriteImage(warped_atlas_ct, str(out_dir / f"{pid}_warped_atlas_{cid}.nii.gz"))

    # Warped labels
    warped_lbl = LabelTransformer.transform_labels(atlas_lbl_eval, eval_fixed_raw, full_composite)

    # Dust cleaning (< 15 voxels)
    cc_filter = sitk.ConnectedComponentImageFilter()
    labeled_img = cc_filter.Execute(warped_lbl)
    relabel_filter = sitk.RelabelComponentImageFilter()
    relabel_filter.SetMinimumObjectSize(15)
    cleaned = relabel_filter.Execute(labeled_img)
    cleaned = sitk.BinaryThreshold(cleaned, lowerThreshold=1, upperThreshold=10000, insideValue=1, outsideValue=0)
    cleaned = sitk.Cast(cleaned, sitk.sitkUInt8)

    # Curvature flow smoothing
    float_img = sitk.Cast(cleaned, sitk.sitkFloat32)
    smoothed_float = sitk.CurvatureFlow(float_img, timeStep=0.0625, numberOfIterations=2)
    smoothed = sitk.BinaryThreshold(smoothed_float, lowerThreshold=0.5, insideValue=1, outsideValue=0)
    smoothed = sitk.Cast(smoothed, sitk.sitkUInt8)

    # Epicardial boundary constraint & coronary tree isolation
    v_arr = sitk.GetArrayFromImage(smoothed) > 0
    p_arr = sitk.GetArrayFromImage(eval_fixed_raw)

    p_myo = (p_arr >= -30) & (p_arr <= 150)
    lbl_m, num_m = ndi.label(p_myo)
    p_myo = (lbl_m == (np.argmax(ndi.sum(p_myo, lbl_m, range(1, num_m + 1))) + 1))
    myo_dist = ndi.distance_transform_edt(~p_myo)

    # Remove rogue voxels > 4.5 mm from myocardium
    v_clean = v_arr & (myo_dist <= 4.5)

    # Retain the primary coronary arterial trees (LCA + RCA)
    lbl_v, num_v = ndi.label(v_clean)
    sizes = [((lbl_v == i).sum(), i) for i in range(1, num_v + 1)]
    sizes.sort(reverse=True)
    if len(sizes) >= 2:
        top_mask = (lbl_v == sizes[0][1]) | (lbl_v == sizes[1][1])
    elif len(sizes) == 1:
        top_mask = (lbl_v == sizes[0][1])
    else:
        top_mask = v_clean

    final_vessel_img = sitk.GetImageFromArray(top_mask.astype(np.uint8))
    final_vessel_img.CopyInformation(warped_lbl)
    sitk.WriteImage(final_vessel_img, str(out_dir / f"{pid}_synthetic_vessel.nii.gz"))

    # Validation metrics
    v_final_arr = sitk.GetArrayFromImage(final_vessel_img) > 0
    c_arr = sitk.GetArrayFromImage(eval_coca_seg) > 0
    myo_dists = myo_dist[v_final_arr]

    edt_ca = ndi.distance_transform_edt(~v_final_arr)
    ca_dists = edt_ca[c_arr] if c_arr.sum() > 0 else np.array([0.0])

    elapsed = time.time() - t_start

    print(f"\n  --- Results for {pid} ---", flush=True)
    print(f"  Atlas Used:             Case {cid}", flush=True)
    print(f"  Max Displacement:       {max_disp:.2f} mm", flush=True)
    print(f"  Min Jacobian det:       {min_jac:.4f}", flush=True)
    print(f"  Vessel Voxels:          {v_final_arr.sum()}", flush=True)
    print(f"  Median Myocardium Dist: {np.median(myo_dists):.2f} mm (Max: {myo_dists.max():.2f} mm)", flush=True)
    print(f"  Median Calcium Dist:    {np.median(ca_dists):.2f} mm (Mean: {ca_dists.mean():.2f} mm)", flush=True)
    print(f"  Calcium within 3.0 mm:  {(ca_dists <= 3.0).mean()*100:.1f}%", flush=True)
    print(f"  Calcium within 5.0 mm:  {(ca_dists <= 5.0).mean()*100:.1f}%", flush=True)
    print(f"  Runtime:                {elapsed:.1f} s", flush=True)

    cohort_summary.append({
        "patient_id": pid,
        "atlas_id": cid,
        "max_disp_mm": max_disp,
        "min_jac": min_jac,
        "vessel_voxels": v_final_arr.sum(),
        "median_myo_dist_mm": np.median(myo_dists),
        "max_myo_dist_mm": myo_dists.max(),
        "median_ca_dist_mm": np.median(ca_dists),
        "mean_ca_dist_mm": ca_dists.mean(),
        "ca_pct_3mm": (ca_dists <= 3.0).mean()*100,
        "ca_pct_5mm": (ca_dists <= 5.0).mean()*100,
        "runtime_s": elapsed
    })

print(f"\n{'='*75}", flush=True)
print("             MULTI-PATIENT VERIFICATION SUMMARY", flush=True)
print(f"{'='*75}", flush=True)
for res in cohort_summary:
    print(f"Patient {res['patient_id']} (Atlas {res['atlas_id']}): "
          f"Med Myo Dist={res['median_myo_dist_mm']:.1f}mm (Max={res['max_myo_dist_mm']:.1f}mm) | "
          f"Med Ca Dist={res['median_ca_dist_mm']:.1f}mm | "
          f"Ca <= 3mm: {res['ca_pct_3mm']:.1f}% | "
          f"Runtime: {res['runtime_s']:.1f}s")
print(f"{'='*75}", flush=True)
