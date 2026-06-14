#!/usr/bin/env python3
r"""
Show XY projection overlaid on floor plan with interactive controls.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import open3d as o3d
from pathlib import Path
import sys
import cv2


def load_transform(transform_path):
    """Load 4x4 transformation matrix from file."""
    if not Path(transform_path).exists():
        print(f"Warning: Transform file not found: {transform_path}")
        return None
    return np.loadtxt(transform_path)


def apply_transform_to_points(points, T):
    """Apply 4x4 transform to Nx3 points."""
    n = points.shape[0]
    ones = np.ones((n, 1))
    pts_h = np.hstack([points, ones])
    transformed = (T @ pts_h.T).T
    return transformed[:, :3]


def filter_central_region(points, percentile=95):
    """Filter points to keep only central region based on distance from centroid."""
    if points.shape[0] == 0:
        return points
    
    centroid = points.mean(axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)
    threshold = np.percentile(distances, percentile)
    mask = distances <= threshold
    
    filtered = points[mask]
    print(f"Filtered: {points.shape[0]} -> {filtered.shape[0]} points ({percentile}% central)")
    return filtered


def project_to_xy(points):
    """Project 3D points to XY plane (drop Z)."""
    return points[:, :2]


def apply_2d_transform(points, scale_x, scale_y, rotation_deg, tx, ty):
    """Apply 2D transform: scale X/Y, rotate, translate."""
    # Scale
    pts = points.copy()
    pts[:, 0] *= scale_x
    pts[:, 1] *= scale_y
    
    # Rotate
    angle_rad = np.radians(rotation_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    pts = (R @ pts.T).T
    
    # Translate
    pts[:, 0] += tx
    pts[:, 1] += ty
    
    return pts


def main():
    # Paths
    colmap_path = r"F:\colmap\first_floor_and_stairs"
    transform_path = r"F:\artifacts\transform.txt"
    floor_plan_path = r"F:\artifacts\1_floor_grid_700x200.png"
    output_path = r"F:\artifacts\transform_2d.txt"
    
    # Load transform
    print(f"Loading transform from {transform_path}...")
    T = load_transform(transform_path)
    
    # Load point cloud
    print(f"\nLoading point cloud from {colmap_path}...")
    scene = o3d.geometry.PointCloud()
    pcd_found = False
    
    folder = Path(colmap_path)
    for ply_file in folder.glob("*.ply"):
        scene = o3d.io.read_point_cloud(str(ply_file))
        pcd_found = True
        break
    
    if not pcd_found:
        print("ERROR: No .ply file found!")
        sys.exit(1)
    
    points = np.asarray(scene.points)
    print(f"Loaded {len(points)} points")
    
    # Apply transform
    if T is not None:
        points = apply_transform_to_points(points, T)
    
    # Filter and project
    points_filtered = filter_central_region(points, percentile=95)
    points_2d = project_to_xy(points_filtered)
    
    proj_w = points_2d[:,0].max() - points_2d[:,0].min()
    proj_h = points_2d[:,1].max() - points_2d[:,1].min()
    print(f"Projection size: {proj_w:.2f} x {proj_h:.2f}")
    
    # Load floor plan
    floor_img = cv2.imread(floor_plan_path)
    floor_img = cv2.cvtColor(floor_img, cv2.COLOR_BGR2RGB)
    fh, fw = floor_img.shape[:2]
    print(f"Floor plan: {fw}x{fh}")

# Initial transform - scale to fit roughly
    scale = min(fw / proj_w, fh / proj_h) * 0.3
    rotation = 0
    tx = fw / 2
    ty = fh / 2
    
    # Create figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111)
    ax.set_title('Floor Plan + Projection (adjust sliders then save)')
    ax.axis('off')
    
    # Sliders - more space for 5 sliders
    plt.subplots_adjust(bottom=0.4)
    ax_scale_x = plt.axes([0.2, 0.32, 0.6, 0.03])
    ax_scale_y = plt.axes([0.2, 0.27, 0.6, 0.03])
    ax_rot = plt.axes([0.2, 0.22, 0.6, 0.03])
    ax_tx = plt.axes([0.2, 0.17, 0.6, 0.03])
    ax_ty = plt.axes([0.2, 0.12, 0.6, 0.03])
    
    s_scale_x = widgets.Slider(ax_scale_x, 'Scale X', 10, 100.0, valinit=scale)
    s_scale_y = widgets.Slider(ax_scale_y, 'Scale Y', 10, 100.0, valinit=scale)
    s_rot = widgets.Slider(ax_rot, 'Rotation', -180, 180, valinit=rotation)
    s_tx = widgets.Slider(ax_tx, 'Translate X', -fw, fw*2, valinit=tx)
    s_ty = widgets.Slider(ax_ty, 'Translate Y', -fh, fh*2, valinit=ty)
    
    # Save button
    ax_save = plt.axes([0.8, 0.05, 0.1, 0.04])
    btn_save = widgets.Button(ax_save, 'Save')
    
    def update(val):
        scale_x = s_scale_x.val
        scale_y = s_scale_y.val
        rotation = s_rot.val
        tx = s_tx.val
        ty = s_ty.val
        
        # Transform points
        pts = apply_2d_transform(points_2d, scale_x, scale_y, rotation, tx, ty)
        
        ax.clear()
        ax.imshow(floor_img)
        ax.scatter(pts[:, 0], pts[:, 1], s=1, c='red', alpha=0.5)
        ax.set_title(f'scale_x={scale_x:.3f}, scale_y={scale_y:.3f}, rot={rotation:.1f}°, tx={tx:.0f}, ty={ty:.0f}')
        ax.axis('off')
        fig.canvas.draw_idle()
    
    s_scale_x.on_changed(update)
    s_scale_y.on_changed(update)
    s_rot.on_changed(update)
    s_tx.on_changed(update)
    s_ty.on_changed(update)
    
    def save(event):
        # Compute where the 3D model origin (0,0) maps to
        origin = np.array([[0.0, 0.0]])
        origin_transformed = apply_2d_transform(origin, s_scale_x.val, s_scale_y.val, s_rot.val, s_tx.val, s_ty.val)
        origin_x_img = origin_transformed[0, 0]
        origin_y_img = origin_transformed[0, 1]
        
        # Convert to floor plan coords (bottom-left = 0,0)
        origin_x_floor = origin_x_img
        origin_y_floor = 200 - origin_y_img
        
        with open(output_path, 'w') as f:
            f.write("# 2D transform: 3D world -> floor plan image (top-left origin)\n")
            f.write("#\n")
            f.write("# Transform parameters (tuned on original image):\n")
            f.write(f"scale_x={s_scale_x.val:.8f}\n")
            f.write(f"scale_y={s_scale_y.val:.8f}\n")
            f.write(f"rotation_deg={s_rot.val:.8f}\n")
            f.write(f"tx={s_tx.val:.8f}\n")
            f.write(f"ty={s_ty.val:.8f}\n")
            f.write(f"original_image_width={fw}\n")
            f.write(f"original_image_height={fh}\n")
            f.write(f"output_width=700\n")
            f.write(f"output_height=200\n")
            f.write("#\n")
            f.write("# 3D model origin (0,0) projects to (in original image space):\n")
            f.write(f"# X={origin_x_img:.8f}, Y={origin_y_img:.8f}\n")
        print(f"Saved to {output_path}")
        print(f"Model origin → Floor plan: X={origin_x_floor:.2f}, Y={origin_y_floor:.2f}")
    
    btn_save.on_clicked(save)
    
    # Initial draw
    update(None)
    plt.show()


if __name__ == '__main__':
    main()