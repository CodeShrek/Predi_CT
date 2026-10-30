import sys
sys.path.insert(0, "src/phase1_segmentation")
from pathlib import Path
import time
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from masking_bspline import Preprocessor, LabelTransformer, get_cardiac_roi, CFG

sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(8)

out_dir = Path("output_bspline_v3")
out_dir.mkdir(parents=True, exist_ok=True)

print("=== Diffeomorphic Multi-Resolution B-Spline Registration ===", flush=True)
t0 = time.time()

# 1. Load data
patient_pack = Preprocessor.run_memory_resample(
    Path("COCA/1cc17f65f909/1cc17f65f909_raw_img.nii.gz"),
    Path("COCA/1cc17f65f909/1cc17f65f909_seg.nii.gz")
)
atlas_pack = Preprocessor.run_memory_resample(
    Path("1-200/194.img.nii.gz"),
    Path("1-200/194.label.nii.gz")
)

fixed_win = patient_pack["reg_win"]
atlas_win = atlas_pack["reg_win"]
eval_fixed_raw = patient_pack["eval_raw"]
eval_coca_seg = patient_pack["eval_lbl"]
atlas_lbl_eval = atlas_pack["eval_lbl"]
atlas_lbl_reg = Preprocessor.resample_volume(atlas_lbl_eval, CFG.REG_SPACING_MM, is_label=True)

# 2. Stage 1: Rigid (Euler3D)
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

# 3. Stage 2: Affine on Cardiac ROI (with Pericardial/Lung Interface)
rigid_lbl = sitk.Resample(atlas_lbl_reg, fixed_win, rigid_tx, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
rigid_arr = sitk.GetArrayFromImage(rigid_lbl)
roi = get_cardiac_roi(rigid_arr, margin_mm=25.0, spacing_mm=CFG.REG_SPACING_MM)
fz0, fz1, fy0, fy1, fx0, fx1 = roi

fixed_arr = sitk.GetArrayFromImage(fixed_win)
mask_arr = np.zeros_like(fixed_arr, dtype=np.uint8)
mask_arr[fz0:fz1, fy0:fy1, fx0:fx1] = 1
metric_mask = sitk.GetImageFromArray(mask_arr)
metric_mask.CopyInformation(fixed_win)

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

# 4. Stage 3: Diffeomorphic B-Spline (learningRate=1.2, relaxation=0.75)
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

# Check Jacobian and displacement
disp_filter = sitk.TransformToDisplacementFieldFilter()
disp_filter.SetReferenceImage(fixed_win)
disp_field = disp_filter.Execute(final_bspline)
disp_arr = sitk.GetArrayFromImage(disp_field)
max_disp = float(np.sqrt((disp_arr**2).sum(axis=-1)).max())

jac_filter = sitk.DisplacementFieldJacobianDeterminantFilter()
jac_img = jac_filter.Execute(disp_field)
jac_arr = sitk.GetArrayFromImage(jac_img)
min_jac = float(jac_arr.min())
max_jac = float(jac_arr.max())
print(f"  Max Displacement: {max_disp:.2f} mm", flush=True)
print(f"  Jacobian det: min={min_jac:.4f}, max={max_jac:.4f} (diffeomorphic: {min_jac > 0})", flush=True)

# Build composite
full_composite = sitk.CompositeTransform(3)
full_composite.AddTransform(affine_composite_tx)
full_composite.AddTransform(final_bspline)

# Warp and save
warped_atlas_ct = sitk.Resample(atlas_pack["eval_raw"], eval_fixed_raw, full_composite, sitk.sitkLinear, -1024.0, sitk.sitkFloat32)
sitk.WriteImage(warped_atlas_ct, str(out_dir / "1cc17f65f909_warped_atlas_194.nii.gz"))

warped_lbl = LabelTransformer.transform_labels(atlas_lbl_eval, eval_fixed_raw, full_composite)
cc_filter = sitk.ConnectedComponentImageFilter()
labeled_img = cc_filter.Execute(warped_lbl)
relabel_filter = sitk.RelabelComponentImageFilter()
relabel_filter.SetMinimumObjectSize(15)
cleaned = relabel_filter.Execute(labeled_img)
cleaned = sitk.BinaryThreshold(cleaned, lowerThreshold=1, upperThreshold=10000, insideValue=1, outsideValue=0)
cleaned = sitk.Cast(cleaned, sitk.sitkUInt8)

float_img = sitk.Cast(cleaned, sitk.sitkFloat32)
smoothed_float = sitk.CurvatureFlow(float_img, timeStep=0.0625, numberOfIterations=2)
smoothed = sitk.BinaryThreshold(smoothed_float, lowerThreshold=0.5, insideValue=1, outsideValue=0)
smoothed = sitk.Cast(smoothed, sitk.sitkUInt8)
smoothed.CopyInformation(warped_lbl)

vessel_path = out_dir / "1cc17f65f909_synthetic_vessel.nii.gz"
sitk.WriteImage(smoothed, str(vessel_path))

v_arr = sitk.GetArrayFromImage(smoothed) > 0
c_arr = sitk.GetArrayFromImage(eval_coca_seg) > 0
from scipy.ndimage import distance_transform_edt
edt_vessel = distance_transform_edt(~v_arr)
ca_dists = edt_vessel[c_arr]
print(f"Mean Distance to Calcium:   {ca_dists.mean():.2f} mm", flush=True)
print(f"Median Distance to Calcium: {np.median(ca_dists):.2f} mm", flush=True)
print(f"Ca within 3.0 mm: {(ca_dists <= 3.0).sum()}/{len(ca_dists)} ({(ca_dists <= 3.0).mean()*100:.1f}%)", flush=True)
print(f"Ca within 5.0 mm: {(ca_dists <= 5.0).sum()}/{len(ca_dists)} ({(ca_dists <= 5.0).mean()*100:.1f}%)", flush=True)
print(f"Total Runtime: {time.time()-t0:.1f} seconds", flush=True)
