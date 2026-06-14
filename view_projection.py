#!/usr/bin/env python3
r"""
Simple 2D projection viewer - shows XY projection of 3D points.

Usage:
    python view_projection.py points3D.ply [--transform transform.txt]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def filter_central_region(points, percentile=95):
    """Filter points to keep only central region based on distance from centroid."""
    if points.shape[0] == 0:
        return points, np.ones(points.shape[0], dtype=bool)
    
    # Remove points with extreme values
    valid_mask = np.all(np.abs(points) < 1e6, axis=1)
    points_clean = points[valid_mask]
    if points_clean.shape[0] != points.shape[0]:
        print(f"Removed {points.shape[0] - points_clean.shape[0]} extreme points")
    
    if points_clean.shape[0] == 0:
        return points, np.ones(points.shape[0], dtype=bool)
    
    centroid = points_clean.mean(axis=0)
    distances = np.linalg.norm(points_clean - centroid, axis=1)
    threshold = np.percentile(distances, percentile)
    mask = distances <= threshold
    
    filtered = points_clean[mask]
    print(f"Filtered: {points_clean.shape[0]} -> {filtered.shape[0]} ({percentile}% central)")
    return filtered, mask


def project_to_xy(points):
    """Project 3D points to XY plane (drop Z)."""
    return points[:, :2]


def load_ply(filepath):
    """Load PLY file."""
    with open(filepath, 'rb') as f:
        header = []
        while True:
            line = f.readline().decode('utf-8').strip()
            header.append(line)
            if line.startswith('end_header'):
                break
        
        vertex_count = 0
        is_binary = False
        for line in header:
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            if 'binary' in line.lower():
                is_binary = True
        
        if is_binary:
            raw = f.read()
            bytes_per_vertex = len(raw) // vertex_count
            if bytes_per_vertex >= 12:
                data = np.frombuffer(raw[:vertex_count*12], dtype=np.float32)
                return data.reshape(-1, 3)
            else:
                data = np.frombuffer(raw, dtype=np.float32)
                return data.reshape(-1, 3)[:, :3]
        else:
            points = []
            for _ in range(vertex_count):
                line = f.readline().decode('utf-8').strip().split()
                points.append([float(x) for x in line[:3]])
            return np.array(points, dtype=np.float32)


def load_transform(filepath):
    """Load 4x4 transform matrix."""
    return np.loadtxt(filepath)


def apply_transform(points, T):
    """Apply 4x4 transform to points."""
    n = points.shape[0]
    ones = np.ones((n, 1))
    pts_h = np.hstack([points, ones])
    transformed = (T @ pts_h.T).T
    return transformed[:, :3]


def main():
    p = argparse.ArgumentParser(description='Show 2D XY projection of 3D points')
    p.add_argument('ply', help='Path to PLY file')
    p.add_argument('--transform', '-t', help='Path to transform matrix (4x4)')
    p.add_argument('--percentile', '-p', type=float, default=95, help='Percentile for filtering')
    args = p.parse_args()

    # Load PLY
    print(f"Loading {args.ply}...")
    points = load_ply(args.ply)
    print(f"Loaded {points.shape[0]} points")
    
    # Apply transform if provided
    if args.transform:
        print(f"Applying transform from {args.transform}...")
        T = load_transform(args.transform)
        points = apply_transform(points, T)
        print(f"Transformed points: {points.shape[0]}")

    # Filter to central region
    points_filtered, _ = filter_central_region(points, percentile=args.percentile)
    
    # Project to XY
    points_2d = project_to_xy(points_filtered)
    
    print(f"\nXY Projection bounds:")
    print(f"  X: [{points_2d[:,0].min():.2f}, {points_2d[:,0].max():.2f}]")
    print(f"  Y: [{points_2d[:,1].min():.2f}, {points_2d[:,1].max():.2f}]")
    print(f"  Range X: {points_2d[:,0].max() - points_2d[:,0].min():.2f}")
    print(f"  Range Y: {points_2d[:,1].max() - points_2d[:,1].min():.2f}")

    # Show projection
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(points_2d[:, 0], points_2d[:, 1], s=0.5, c='blue', alpha=0.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'XY Projection ({len(points_2d)} points)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()