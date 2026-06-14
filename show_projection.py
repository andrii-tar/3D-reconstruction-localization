#!/usr/bin/env python3
r"""
Show XY projection of 3D model - using same approach as ace_loc.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from pathlib import Path
import sys


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


def main():
    # Paths - same as ace_loc.py
    colmap_path = r"F:\colmap\first_floor_and_stairs"
    transform_path = r"F:\artifacts\transform.txt"
    
    # Load transform
    print(f"Loading transform from {transform_path}...")
    T = load_transform(transform_path)
    if T is not None:
        print(f"Transform:\n{T}")
    
    # Load point cloud using Open3D (same as ace_loc.py)
    print(f"\nLoading point cloud from {colmap_path}...")
    scene = o3d.geometry.PointCloud()
    pcd_found = False
    
    folder = Path(colmap_path)
    for ply_file in folder.glob("*.ply"):
        print(f"Found: {ply_file.name}")
        scene = o3d.io.read_point_cloud(str(ply_file))
        pcd_found = True
        break
    
    if not pcd_found:
        print("ERROR: No .ply file found!")
        sys.exit(1)
    
    # Get points as numpy array
    points = np.asarray(scene.points)
    print(f"Loaded {len(points)} points")
    
    # Apply transform
    if T is not None:
        print("Applying transform...")
        points = apply_transform_to_points(points, T)
    
    # Filter to central 95%
    points_filtered = filter_central_region(points, percentile=95)
    
    # Project to XY
    points_2d = project_to_xy(points_filtered)
    
    print(f"\nXY range:")
    print(f"  X: [{points_2d[:,0].min():.2f}, {points_2d[:,0].max():.2f}]")
    print(f"  Y: [{points_2d[:,1].min():.2f}, {points_2d[:,1].max():.2f}]")
    
    # Show 2D projection (same as ace_loc.py visualize_2d_projection)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(points_2d[:, 0], points_2d[:, 1], s=0.5, c='blue', alpha=0.3, label='Scene Points')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('XY Projection (Top View)')
    ax.axis('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()