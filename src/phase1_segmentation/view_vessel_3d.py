"""
3D Vessel Mask Viewer — Renders the Phase 1 vessel scaffold as an interactive 3D surface.
Uses PyVista + VTK for a proper interactive 3D window with rotation, zoom, etc.

Usage:
    python view_vessel_3d.py --mask output/1cc17f65f909_vessels_best.nii.gz
    python view_vessel_3d.py --mask output/1cc17f65f909_vessels_best.nii.gz --save render.png
    python view_vessel_3d.py --mask output/1cc17f65f909_vessels_best.nii.gz --ess ess_predictions.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import pyvista as pv


def load_mask_as_mesh(mask_path, smooth=True):
    """Load a NIfTI binary mask and convert to a PyVista surface mesh."""
    print(f"Loading mask: {mask_path}")
    img = sitk.ReadImage(str(mask_path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    spacing = np.array(img.GetSpacing())       # (X, Y, Z)
    origin  = np.array(img.GetOrigin())        # (X, Y, Z)

    n_voxels = int(np.count_nonzero(arr))
    print(f"  Shape (ZYX): {arr.shape}")
    print(f"  Spacing: {spacing} mm")
    print(f"  Vessel voxels: {n_voxels}")

    if n_voxels == 0:
        print("ERROR: Mask is empty.")
        sys.exit(1)

    # Create PyVista uniform grid (note: pyvista uses XYZ ordering)
    grid = pv.ImageData(
        dimensions=(arr.shape[2] + 1, arr.shape[1] + 1, arr.shape[0] + 1),
        spacing=tuple(spacing),
        origin=tuple(origin)
    )
    # Transpose ZYX -> XYZ flattened (Fortran order)
    grid.cell_data["vessel"] = arr.T.flatten()

    # Extract isosurface at threshold 0.5
    grid = grid.cell_data_to_point_data()
    surface = grid.contour(isosurfaces=[0.5], scalars="vessel")

    if smooth:
        surface = surface.smooth(n_iter=50, relaxation_factor=0.1)

    print(f"  Mesh: {surface.n_points} vertices, {surface.n_cells} faces")
    return surface, n_voxels


def render_vessel_3d(mask_path, smooth=True, save_path=None, ess_csv=None):
    """Render the vessel mask as an interactive 3D surface."""
    surface, n_voxels = load_mask_as_mesh(mask_path, smooth)

    # Set up the plotter
    plotter = pv.Plotter(window_size=[1400, 1000], off_screen=bool(save_path))
    plotter.set_background("black", top="midnightblue")

    # If ESS data provided, color the surface by ESS values
    if ess_csv and Path(ess_csv).exists():
        import pandas as pd
        print(f"  Loading ESS data from {ess_csv}...")
        df = pd.read_csv(ess_csv)
        ess_points = df[['x', 'y', 'z']].values
        ess_values = df['ess_magnitude'].values

        # Map ESS values onto surface vertices via nearest-neighbour
        from scipy.spatial import cKDTree
        tree = cKDTree(ess_points)
        _, idx = tree.query(surface.points)
        surface["ESS (Pa)"] = ess_values[idx]

        plotter.add_mesh(surface, scalars="ESS (Pa)", cmap="coolwarm",
                         clim=[0, 5], opacity=0.85, show_edges=False,
                         smooth_shading=True,
                         scalar_bar_args={"title": "ESS (Pa)", "color": "white"})
    else:
        plotter.add_mesh(surface, color="crimson", opacity=0.7,
                         show_edges=False, smooth_shading=True,
                         specular=0.5, specular_power=15)

    plotter.add_text(
        f"Phase 1 Vessel Scaffold\n{Path(mask_path).stem}\n{n_voxels} voxels",
        position="upper_left", font_size=12, color="white"
    )
    plotter.add_axes(color="white")

    if save_path:
        plotter.show(screenshot=save_path, auto_close=False)
        print(f"  Screenshot saved to: {save_path}")
    else:
        plotter.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Vessel Mask Viewer (PyVista)")
    parser.add_argument("--mask", type=str, required=True,
                        help="Path to vessel mask NIfTI (.nii.gz)")
    parser.add_argument("--ess", type=str, default=None,
                        help="Optional ESS CSV to color the surface by shear stress")
    parser.add_argument("--save", type=str, default=None,
                        help="Save screenshot to PNG instead of interactive window")
    parser.add_argument("--no-smooth", action="store_true",
                        help="Skip surface smoothing")
    args = parser.parse_args()

    render_vessel_3d(args.mask, smooth=not args.no_smooth,
                     save_path=args.save, ess_csv=args.ess)
