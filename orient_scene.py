#!/usr/bin/env python3
r"""
orient_scene.py

Load a COLMAP sparse reconstruction, estimate the best-fit plane of camera
centers (expected circular panorama trajectories), and rotate the scene so
that that plane is parallel to the chosen ground axis (default Z-up).

Outputs:
 - 4x4 transform matrix saved as transform.txt
 - transformed sparse points saved as transformed_points.ply (ASCII)
 - transformed camera centers saved as camera_centers_transformed.csv
 - before/after visualization saved as orient_before_after.png (optional)

Usage examples:
  python orient_scene.py path/to/reconstruction --output_dir path/to/out --show

Requirements: pycolmap, numpy, matplotlib
"""

import argparse
import pathlib
import sys
import numpy as np
import math

try:
    import pycolmap
except Exception:
    print("Error: pycolmap not installed. Please install pycolmap to run this script.")
    raise

import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def get_camera_poses(reconstruction):
    """Return list of (image_id, name, center, forward, up, right)"""
    cameras_data = []
    failed = 0
    
    for img_id, img in reconstruction.images.items():
        C = None
        Rcw = None
        
        # Check has_pose first (new pycolmap API)
        if not getattr(img, 'has_pose', False):
            failed += 1
            continue
        
        # Get rotation matrix
        cfw = getattr(img, 'cam_from_world', None)
        if cfw is not None:
            try:
                pose = cfw() if callable(cfw) else cfw
                M = np.array(pose.matrix())
                if M.shape == (3, 4):
                    Rcw = M[:3, :3]
                    t = M[:3, 3]
                    C = -Rcw.T @ t
                elif M.shape == (4, 4):
                    Rcw = M[:3, :3]
                    t = M[:3, 3]
                    C = -Rcw.T @ t
            except Exception:
                pass
        
        # Fallback to qvec/tvec
        if Rcw is None:
            try:
                qvec = getattr(img, 'qvec', None)
                tvec = getattr(img, 'tvec', None)
                if qvec is not None and tvec is not None:
                    qvec = np.array(qvec)
                    tvec = np.array(tvec)
                    if hasattr(pycolmap, 'qvec2rotmat'):
                        Rcw = pycolmap.qvec2rotmat(qvec)
                        C = -Rcw.T @ tvec
            except Exception:
                pass
        
        if Rcw is None or C is None:
            failed += 1
            continue
        
        # Camera frame: columns of Rcw are [right, up, forward] in world coords
        # forward = -Rcw[2,:] (camera looks along -Z in its local frame)
        # up = Rcw[1,:]
        # right = Rcw[0,:]
        forward = -Rcw[2, :]
        up = Rcw[1, :]
        right = Rcw[0, :]
        
        name = img.name if hasattr(img, 'name') else str(img_id)
        cameras_data.append((img_id, name, np.asarray(C), forward, up, right))
    
    if failed > 0:
        print(f"Note: {failed} images without valid pose")
    return cameras_data


def fit_plane_svd(points):
    """Fit plane to points using SVD. Returns plane normal (unit) and centroid."""
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 3:
        raise ValueError('Need at least 3 points to fit a plane')
    centroid = pts.mean(axis=0)
    X = pts - centroid
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    normal = Vt[-1, :]
    # Ensure unit
    normal = normal / np.linalg.norm(normal)
    return normal, centroid


def rotation_matrix_from_vectors(a, b):
    """Compute rotation matrix that rotates vector a to vector b.
    Both a and b are 3-element arrays.
    Handles parallel and anti-parallel cases.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.allclose(a, 0) or np.allclose(b, 0):
        raise ValueError('Zero vector provided')
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if np.allclose(v, 0):
        # Vectors are parallel or anti-parallel
        if c > 0:
            return np.eye(3)
        else:
            # 180 degree rotation around any orthogonal axis
            # Find orthogonal vector
            orth = np.array([1.0, 0.0, 0.0])
            if np.allclose(np.abs(a), orth):
                orth = np.array([0.0, 1.0, 0.0])
            axis = np.cross(a, orth)
            axis = axis / np.linalg.norm(axis)
            return rotation_matrix_axis_angle(axis, math.pi)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return R


def filter_central_region(points, percentile=95):
    """Keep only central region based on distance from centroid.
    
    Args:
        points: Nx3 array of 3D points
        percentile: Keep points within this percentile of distance (default 95)
    
    Returns:
        filtered_points: Points within the central region
        mask: Boolean mask of kept points
    """
    if points.shape[0] == 0:
        return points, np.ones(points.shape[0], dtype=bool)
    
    # Compute centroid
    centroid = points.mean(axis=0)
    
    # Compute distances from centroid
    distances = np.linalg.norm(points - centroid, axis=1)
    
    # Keep points within percentile distance
    threshold = np.percentile(distances, percentile)
    mask = distances <= threshold
    
    filtered = points[mask]
    print(f"Filtered: {points.shape[0]} -> {filtered.shape[0]} points (keeping {percentile}% central region)")
    
    return filtered, mask


def rotation_matrix_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    a = math.cos(angle / 2.0)
    b, c, d = -axis * math.sin(angle / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def write_ply_ascii(points, path):
    points = np.asarray(points, dtype=float)
    n = points.shape[0]
    header = f"""ply
format ascii 1.0
element vertex {n}
property float x
property float y
property float z
end_header
"""
    with open(path, 'w') as f:
        f.write(header)
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")


def main():
    p = argparse.ArgumentParser(description='Orient scene so camera trajectories are parallel to ground')
    p.add_argument('reconstruction', type=pathlib.Path, help='Path to COLMAP reconstruction folder (cameras.bin, images.bin, points3D.bin)')
    p.add_argument('--output_dir', type=pathlib.Path, default=None, help='Directory to save outputs (default: reconstruction parent/dense)')
    p.add_argument('--up', choices=['z','y','x'], default='z', help='Which axis is up after orientation (default z)')
    p.add_argument('--set_ground', action='store_true', help='Translate the scene so the minimum Z is 0 after rotation')
    p.add_argument('--downsample', type=float, default=1.0, help='Fraction of points to include in output PLY (0-1)')
    p.add_argument('--show', action='store_true', help='Save a before/after visualization and show it')
    args = p.parse_args()

    rec_path = args.reconstruction
    if not rec_path.exists():
        print('Reconstruction path does not exist:', rec_path)
        sys.exit(2)

    # Determine default output dir
    out_dir = args.output_dir
    if out_dir is None:
        out_dir = rec_path.parent
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('Loading reconstruction from', rec_path)
    reconstruction = pycolmap.Reconstruction(str(rec_path))

    cameras_data = get_camera_poses(reconstruction)
    if len(cameras_data) < 3:
        print('Need at least 3 cameras to estimate orientation. Found:', len(cameras_data))
        sys.exit(3)
    
    # Extract forward and up vectors
    forwards = np.array([c[3] for c in cameras_data])
    ups = np.array([c[4] for c in cameras_data])
    centers = np.array([c[2] for c in cameras_data])
    
    print(f'Using {len(cameras_data)} cameras')
    print(f'Forward vectors sample: {forwards[:3]}')
    print(f'Up vectors sample: {ups[:3]}')
    
    # Method: Use up vectors to find floor normal
    # For indoor scenes, camera up vectors point roughly toward ceiling
    # The floor normal is approximately -average(up) or the orthogonal direction
    
    # Option 1: Use SVD on up vectors to find the dominant vertical direction
    up_mean = np.mean(ups, axis=0)
    up_mean = up_mean / np.linalg.norm(up_mean)
    print(f'Mean up vector: {up_mean}')
    
    # Option 2: SVD on forward vectors - they lie in plane parallel to floor
    # The normal to this plane is the floor normal
    f_valid = ~np.isnan(forwards).any(axis=1)
    forwards_clean = forwards[f_valid]
    
    # SVD on forward vectors
    f_centroid = forwards_clean.mean(axis=0)
    f_centered = forwards_clean - f_centroid
    U, S, Vt = np.linalg.svd(f_centered, full_matrices=False)
    forward_normal = Vt[-1, :]  # Normal to the plane formed by forward vectors
    forward_normal = forward_normal / np.linalg.norm(forward_normal)
    print(f'Floor normal from forward vectors: {forward_normal}')
    
    # Use the forward-derived normal (more robust for non-panorama scenes)
    normal = forward_normal
    centroid = centers.mean(axis=0)

    # Desired up vector
    up_map = {'z': np.array([0.0, 0.0, 1.0]), 'y': np.array([0.0, 1.0, 0.0]), 'x': np.array([1.0, 0.0, 0.0])}
    target_up = up_map[args.up]

    # Compute rotation matrix that aligns normal -> target_up
    R = rotation_matrix_from_vectors(normal, target_up)
    print('Rotation matrix to align plane normal to', args.up, 'axis:\n', R)

    # Build 4x4 transform (rotation, no translation yet)
    T = np.eye(4)
    T[:3, :3] = R

    # Collect points
    pts3 = []
    for pid, p in reconstruction.points3D.items():
        pts3.append(np.array(p.xyz, dtype=float))
    pts3 = np.vstack(pts3) if len(pts3) > 0 else np.zeros((0, 3))

    # Apply rotation
    pts_rot = (R.dot(pts3.T)).T if pts3.shape[0] > 0 else pts3
    cams_rot = (R.dot(centers.T)).T

    # Optionally translate so min Z = 0
    trans = np.zeros(3)
    if args.set_ground:
        all_z = np.concatenate([pts_rot[:, 2] if pts_rot.shape[0] > 0 else np.array([]), cams_rot[:, 2]])
        if all_z.size > 0:
            min_z = np.min(all_z)
            trans[2] = -min_z
            print('Applying vertical translation to set ground at Z=0: shift Z by', trans[2])
    # Apply translation
    pts_trans = pts_rot + trans
    cams_trans = cams_rot + trans

    # Update 4x4 transform
    T[:3, 3] = trans

    # Save transform
    tf_path = out_dir / 'transform.txt'
    np.savetxt(tf_path, T, fmt='%.6f')
    print('Saved transform to', tf_path)

    # Save transform info for later use
    info_path = out_dir / 'transform_info.txt'
    with open(info_path, 'w') as f:
        f.write(f"# Transformation matrix (4x4)\n")
        f.write(f"# Use this to transform points: T @ [x, y, z, 1]\n")
        f.write(f"\n# Full 4x4 transform:\n")
        for row in T:
            f.write(' '.join(f'{v:.8f}' for v in row) + '\n')
        f.write(f"\n# Rotation (3x3):\n")
        for row in R:
            f.write(' '.join(f'{v:.8f}' for v in row) + '\n')
        f.write(f"\n# Translation: {trans[0]:.8f} {trans[1]:.8f} {trans[2]:.8f}\n")
        f.write(f"\n# Z offset (for floor alignment): {trans[2]:.8f}\n")
        f.write(f"\n# Floor normal (estimated): {normal[0]:.8f} {normal[1]:.8f} {normal[2]:.8f}\n")
    print('Saved transform info to', info_path)

    # Downsample points for PLY
    if pts_trans.shape[0] > 0 and (args.downsample > 0 and args.downsample < 1.0):
        rng = np.random.default_rng(42)
        idx = rng.choice(pts_trans.shape[0], size=int(np.ceil(args.downsample * pts_trans.shape[0])), replace=False)
        pts_out = pts_trans[idx]
    else:
        pts_out = pts_trans

    ply_path = out_dir / 'transformed_points.ply'
    write_ply_ascii(pts_out, ply_path)
    print('Saved transformed points (PLY) to', ply_path)

    # Visualization: 2D XY projection before/after
    if args.show:
        # Filter points for cleaner visualization
        pts3_filtered, mask_before = filter_central_region(pts3, percentile=95)
        centers_filtered, _ = filter_central_region(centers, percentile=95)
        
        pts_trans_filtered, mask_after = filter_central_region(pts_trans, percentile=95)
        cams_trans_filtered, _ = filter_central_region(cams_trans, percentile=95)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Before (XY projection)
        ax1 = axes[0]
        ax1.set_title('BEFORE - XY Projection (Top View, 95% central)')
        if pts3_filtered.shape[0] > 0:
            ax1.scatter(pts3_filtered[:, 0], pts3_filtered[:, 1], s=0.5, c='blue', alpha=0.3, label='Points')
        if centers_filtered.shape[0] > 0:
            ax1.scatter(centers_filtered[:, 0], centers_filtered[:, 1], s=5, c='red', alpha=0.8, label='Cameras')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.axis('equal')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # After (XY projection)
        ax2 = axes[1]
        ax2.set_title('AFTER - XY Projection (Top View, 95% central)')
        if pts_trans_filtered.shape[0] > 0:
            ax2.scatter(pts_trans_filtered[:, 0], pts_trans_filtered[:, 1], s=0.5, c='blue', alpha=0.3, label='Points')
        if cams_trans_filtered.shape[0] > 0:
            ax2.scatter(cams_trans_filtered[:, 0], cams_trans_filtered[:, 1], s=5, c='red', alpha=0.8, label='Cameras')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.axis('equal')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        out_viz = out_dir / 'orient_before_after.png'
        plt.savefig(out_viz, dpi=200)
        print('Saved before/after visualization to', out_viz)
        plt.show()


if __name__ == '__main__':
    main()

