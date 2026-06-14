#!/usr/bin/env python3
r"""
visualize_reconstruction.py

Load a COLMAP sparse reconstruction using pycolmap and visualize
sparse 3D points and camera positions/orientations using matplotlib.

Usage examples (PowerShell):
  python visualize_reconstruction.py "D:/data/colmap_model" --show
  python visualize_reconstruction.py "D:/data/colmap_model" --downsample 0.1 --output view.png

Requirements: pycolmap, numpy, matplotlib
"""

import argparse
import pathlib
import sys
import random

import numpy as np

try:
    import pycolmap
except Exception as e:
    print("Error: failed to import pycolmap. Make sure pycolmap is installed.")
    print("See pycolmap README: https://pypi.org/project/pycolmap/")
    raise

import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (required for 3d projection)


def parse_args():
    p = argparse.ArgumentParser(description="Visualize COLMAP sparse reconstruction")
    p.add_argument("reconstruction", type=pathlib.Path, help="Path to COLMAP reconstruction directory (cameras.bin, images.bin, points3D.bin)")
    p.add_argument("--downsample", type=float, default=1.0, help="Fraction of points to keep (0-1). Default=1.0 (all points)")
    p.add_argument("--point-size", type=float, default=1.0, help="Point size for scatter plot")
    p.add_argument("--camera-scale", type=float, default=0.5, help="Scale factor for camera axes/arrows")
    p.add_argument("--output", type=pathlib.Path, default=None, help="If set, save a PNG to this path")
    p.add_argument("--show", action="store_true", help="Show interactive matplotlib window (requires suitable backend)")
    return p.parse_args()


def compute_camera_center_and_axes(image):
    """
    Given a pycolmap Image object, return the camera center (3,) and axes (3,3)
    where axes[:, i] is the i-th axis vector in world coordinates.

    Uses pycolmap.qvec2rotmat to convert quaternion to rotation matrix.
    Computes camera center C = -R.T @ tvec.
    The camera axes are the columns of R.T (they represent camera frame axes in world coords).
    """
    # Support multiple pycolmap Image layouts across versions:
    # 1) image.qvec / image.tvec (quaternion + translation)
    # 2) image.projection_center() -> returns camera center directly
    # 3) image.cam_from_world -> 3x4 or 4x4 transform (camera-from-world)
    # We'll try these in order.
    # 1) qvec/tvec
    if hasattr(image, 'qvec') and getattr(image, 'qvec') is not None:
        qvec = np.array(image.qvec)
        tvec = np.array(image.tvec)
        R = pycolmap.qvec2rotmat(qvec)  # rotation matrix
        C = -R.T.dot(tvec)
        axes = R.T
        return C, axes

    # 2) projection_center() method/property
    proj = getattr(image, 'projection_center', None)
    if proj is not None:
        try:
            if callable(proj):
                C = np.array(proj())
            else:
                C = np.array(proj)
        except Exception:
            C = None
        if C is not None and C.shape == (3,):
            # Try to get rotation from cam_from_world if available
            cf = getattr(image, 'cam_from_world', None)
            if cf is not None:
                try:
                    M = np.array(cf)
                    if M.shape[0] >= 3 and M.shape[1] >= 3:
                        Rcw = M[:3, :3]
                        axes = Rcw.T
                        return C, axes
                except Exception:
                    pass
            # Fallback: no orientation available -> axes = identity
            return C, np.eye(3)

    # 3) cam_from_world (directly extract center and rotation)
    cf = getattr(image, 'cam_from_world', None)
    if cf is not None:
        try:
            M = np.array(cf)
            if M.shape == (4, 4):
                Rcw = M[:3, :3]
                t = M[:3, 3]
                C = -Rcw.T.dot(t)
                axes = Rcw.T
                return C, axes
            # sometimes stored as 3x4
            if M.shape[0] == 3 and M.shape[1] == 4:
                Rcw = M[:3, :3]
                t = M[:3, 3]
                C = -Rcw.T.dot(t)
                axes = Rcw.T
                return C, axes
        except Exception:
            pass

    raise RuntimeError('Unable to extract camera center and axes from image object')


def main():
    args = parse_args()
    rec_path = args.reconstruction
    if not rec_path.exists():
        print(f"Error: reconstruction path '{rec_path}' does not exist.")
        sys.exit(2)

    # If running headless and user wants to save, use Agg backend
    if args.output and not args.show:
        matplotlib.use("Agg")

    print(f"Loading reconstruction from: {rec_path}")
    reconstruction = pycolmap.Reconstruction(str(rec_path))

    # Collect points
    pts3 = []
    for pid, p in reconstruction.points3D.items():
        # p.xyz is a 3-element iterable
        pts3.append(np.array(p.xyz, dtype=float))

    if len(pts3) == 0:
        print("Warning: reconstruction contains no 3D points to visualize.")
        pts = np.zeros((0, 3))
    else:
        pts = np.vstack(pts3)

    # Downsample points if requested
    if args.downsample <= 0 or args.downsample > 1:
        print("--downsample must be in (0, 1]. Using 1.0")
        ds = 1.0
    else:
        ds = args.downsample

    if pts.shape[0] > 0 and ds < 1.0:
        rng = np.random.default_rng(seed=42)
        keep_mask = rng.choice(pts.shape[0], size=int(np.ceil(ds * pts.shape[0])), replace=False)
        pts = pts[keep_mask]

    # Collect camera centers and axes
    cam_centers = []
    cam_axes = []
    cam_names = []
    for img_id, img in reconstruction.images.items():
        try:
            C, axes = compute_camera_center_and_axes(img)
            cam_centers.append(C)
            cam_axes.append(axes)
            cam_names.append(img.name if hasattr(img, 'name') else str(img_id))
        except Exception as e:
            print(f"Warning: failed to compute camera pose for image id {img_id}: {e}")

    cam_centers = np.vstack(cam_centers) if len(cam_centers) > 0 else np.zeros((0, 3))

    # Plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('COLMAP sparse reconstruction (points + camera centers)')

    # Plot points
    if pts.shape[0] > 0:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=max(args.point_size, 0.1), c='k', marker='.', alpha=0.6)

    # Plot camera centers as colored points and axes
    if cam_centers.shape[0] > 0:
        ax.scatter(cam_centers[:, 0], cam_centers[:, 1], cam_centers[:, 2], c='r', s=15, marker='^', label='camera centers')

        # Draw axes for each camera
        scale = float(args.camera_scale)
        for i in range(cam_centers.shape[0]):
            C = cam_centers[i]
            axes = cam_axes[i]
            # axes columns: axes[:,0], axes[:,1], axes[:,2]
            x_axis = axes[:, 0] * scale
            y_axis = axes[:, 1] * scale
            z_axis = axes[:, 2] * scale
            # draw as lines
            ax.plot([C[0], C[0] + x_axis[0]], [C[1], C[1] + x_axis[1]], [C[2], C[2] + x_axis[2]], c='r')
            ax.plot([C[0], C[0] + y_axis[0]], [C[1], C[1] + y_axis[1]], [C[2], C[2] + y_axis[2]], c='g')
            ax.plot([C[0], C[0] + z_axis[0]], [C[1], C[1] + z_axis[1]], [C[2], C[2] + z_axis[2]], c='b')

    # Axes labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set equal aspect ratio roughly by scaling axes to point cloud extents
    def set_equal_axis(ax, pts_all):
        if pts_all.shape[0] == 0:
            return
        xmin, ymin, zmin = pts_all.min(axis=0)
        xmax, ymax, zmax = pts_all.max(axis=0)
        xmid = 0.5 * (xmin + xmax)
        ymid = 0.5 * (ymin + ymax)
        zmid = 0.5 * (zmin + zmax)
        maxrange = max(xmax - xmin, ymax - ymin, zmax - zmin) * 0.6 + 1e-6
        ax.set_xlim(xmid - maxrange / 2, xmid + maxrange / 2)
        ax.set_ylim(ymid - maxrange / 2, ymid + maxrange / 2)
        ax.set_zlim(zmid - maxrange / 2, zmid + maxrange / 2)

    # Combine points and camera centers for bounds
    all_pts = pts
    if cam_centers.shape[0] > 0:
        all_pts = np.vstack([pts, cam_centers]) if pts.shape[0] > 0 else cam_centers

    if all_pts.shape[0] > 0:
        set_equal_axis(ax, all_pts)

    ax.legend()
    plt.tight_layout()

    if args.output:
        outp = str(args.output)
        print(f"Saving figure to {outp}")
        plt.savefig(outp, dpi=200)

    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
