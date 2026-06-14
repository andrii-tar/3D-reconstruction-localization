#!/usr/bin/env python3
r"""
visualize_ply.py

Load a .ply point cloud, apply a transformation matrix, and show 3D preview.

Usage:
    python visualize_ply.py pointcloud.ply transform.txt [--point_size 1]
"""

import argparse
import numpy as np
import sys


def load_ply(filepath):
    """Load PLY file (ASCII or binary). Returns points as Nx3 array."""
    with open(filepath, 'rb') as f:
        header = []
        while True:
            line = f.readline().decode('utf-8').strip()
            header.append(line)
            if line.startswith('end_header'):
                break

        # Parse header
        vertex_count = 0
        is_binary = False
        for line in header:
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            if 'binary' in line.lower():
                is_binary = True

        print(f"PLY: {vertex_count} vertices, format={'binary' if is_binary else 'ascii'}")

        if is_binary:
            # Binary format - read all bytes and try to extract XYZ
            raw = f.read()
            total_bytes = len(raw)
            bytes_per_vertex = total_bytes // vertex_count
            print(f"Binary PLY: {total_bytes} bytes, ~{bytes_per_vertex} bytes/vertex")
            
            # Try different layouts
            if bytes_per_vertex >= 12:
                # Likely x,y,z (3 floats = 12 bytes)
                data = np.frombuffer(raw[:vertex_count*12], dtype=np.float32)
                points = data.reshape(-1, 3)
            elif bytes_per_vertex >= 6:
                # Maybe x,y,z as doubles
                data = np.frombuffer(raw[:vertex_count*8], dtype=np.float64)
                points = data.reshape(-1, 3)
            else:
                # Might have colors too
                floats_per_vertex = bytes_per_vertex // 4
                data = np.frombuffer(raw, dtype=np.float32)
                points = data[:, :3] if floats_per_vertex >= 3 else data.reshape(-1, floats_per_vertex)[:, :3]
        else:
            # ASCII format
            points = []
            for _ in range(vertex_count):
                line = f.readline().decode('utf-8').strip().split()
                points.append([float(x) for x in line[:3]])
            points = np.array(points, dtype=np.float32)

    return points


def load_transform(filepath):
    """Load 4x4 transformation matrix from file."""
    T = np.loadtxt(filepath)
    
    # Validate transform
    R = T[:3, :3]
    if R.shape == (3, 3):
        # Check orthonormality
        det = np.linalg.det(R)
        orthogonal = np.allclose(R @ R.T, np.eye(3), atol=1e-4)
        print(f"Transform validation: det={det:.4f}, orthogonal={orthogonal}")
        if not orthogonal:
            print("WARNING: Rotation matrix is not orthonormal!")
            print("R @ R.T should be identity:")
            print(R @ R.T)
    return T


def apply_transform(points, T):
    """Apply 4x4 transformation to points (Nx3)."""
    n = points.shape[0]
    ones = np.ones((n, 1))
    pts_h = np.hstack([points, ones])
    transformed = (T @ pts_h.T).T
    return transformed[:, :3]


def main():
    p = argparse.ArgumentParser(description='Visualize PLY with transformation')
    p.add_argument('ply', type=str, help='Path to .ply file')
    p.add_argument('transform', type=str, help='Path to transformation matrix (4x4)')
    p.add_argument('--point_size', type=float, default=1.0, help='Point size for visualization')
    p.add_argument('--output', type=str, default=None, help='Save screenshot to file')
    args = p.parse_args()

    print(f"Loading PLY from {args.ply}")
    points = load_ply(args.ply)
    print(f"Loaded {points.shape[0]} points")
    print(f"Points range (before NaN check): [{points.min()}, {points.max()}]")
    nan_count = np.isnan(points).sum()
    if nan_count > 0:
        print(f"WARNING: {nan_count} NaN values in points, removing them")
        points = points[~np.isnan(points).any(axis=1)]
        print(f"After cleanup: {points.shape[0]} points")

    print(f"Loading transform from {args.transform}")
    T = load_transform(args.transform)
    print(f"Transform:\n{T}")

    points_transformed = apply_transform(points, T)
    print(f"Transformed points range: [{points_transformed.min():.2f}, {points_transformed.max():.2f}]")

    try:
        import open3d as o3d
        print("\nUsing Open3D for visualization")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_transformed)

        # Create coordinate frame
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

        o3d.visualization.draw_geometries([pcd, coord], 
            window_name="PLY Viewer")
        
    except ImportError:
        print("\nOpen3D not available, using Matplotlib")
        import matplotlib
        matplotlib.use('TkAgg')
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(points_transformed[:, 0], points_transformed[:, 1], points_transformed[:, 2], 
                   s=args.point_size, c='blue', alpha=0.5)
        
        # Add coordinate axes
        ax.quiver(0, 0, 0, 1, 0, 0, color='r', arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, 0, 1, 0, color='g', arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, 0, 0, 1, color='b', arrow_length_ratio=0.1)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"Transformed PLY: {args.ply}")
        
        if args.output:
            plt.savefig(args.output, dpi=150)
            print(f"Saved to {args.output}")
        else:
            plt.show()


if __name__ == '__main__':
    main()