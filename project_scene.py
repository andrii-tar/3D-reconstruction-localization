#!/usr/bin/env python3
r"""
project_scene.py

Load a COLMAP reconstruction (pycolmap), reuse orientation logic to align the
camera-trajectory plane to a chosen up axis, rotate the scene, and produce a
plan (top-down) density image that highlights vertical structures (walls)
while ignoring floor/ceiling by selecting mid-height bins.

Outputs:
 - density image PNG (plan view)
 - overlay PNG with camera centers
 - CSV of projected 2D points (x,y,z and pixel coords)
 - optional PLY of rotated points
 - transform.txt (4x4) used to rotate/translate the scene

Usage example:
  python project_scene.py colmap_projects/flower_5fps/dense --output_dir view_or --grid_res 0.02 --height_bins 48 --min_height -1 --max_height 1.5 --ignore_low_frac 0.05 --ignore_high_frac 0.05 --downsample 0.5 --save_ply --show

Requirements: pycolmap, numpy, matplotlib
"""

import argparse
import pathlib
import sys
import numpy as np
import math
import csv

try:
    import pycolmap
except Exception:
    print("Error: pycolmap not installed. Please install pycolmap to run this script.")
    raise

# Reuse utility functions from orient_scene.py where possible
try:
    from orient_scene import get_camera_centers, fit_plane_svd, rotation_matrix_from_vectors, write_ply_ascii
except Exception:
    # If import fails, define lightweight fallbacks/copies (only what's needed)
    def get_camera_centers(reconstruction):
        centers = []
        for img_id, img in reconstruction.images.items():
            C = None
            if hasattr(img, 'qvec') and getattr(img, 'qvec') is not None:
                qvec = np.array(img.qvec)
                tvec = np.array(img.tvec)
                if hasattr(pycolmap, 'qvec2rotmat'):
                    R = pycolmap.qvec2rotmat(qvec)
                    C = (-R.T).dot(tvec)
            if C is None and hasattr(img, 'projection_center'):
                try:
                    pc = img.projection_center
                    C = np.array(pc() if callable(pc) else pc)
                except Exception:
                    C = None
            if C is None and hasattr(img, 'cam_from_world'):
                try:
                    M = np.array(img.cam_from_world)
                    if M.shape == (4, 4):
                        Rcw = M[:3, :3]
                        t = M[:3, 3]
                        C = -Rcw.T.dot(t)
                    elif M.shape[0] == 3 and M.shape[1] == 4:
                        Rcw = M[:3, :3]
                        t = M[:3, 3]
                        C = -Rcw.T.dot(t)
                except Exception:
                    C = None
            if C is None:
                print(f"Warning: couldn't get camera center for image id {img_id}")
                continue
            name = img.name if hasattr(img, 'name') else str(img_id)
            centers.append((img_id, name, np.asarray(C, dtype=float)))
        return centers

    def fit_plane_svd(points):
        pts = np.asarray(points, dtype=float)
        if pts.shape[0] < 3:
            raise ValueError('Need at least 3 points to fit a plane')
        centroid = pts.mean(axis=0)
        X = pts - centroid
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        normal = Vt[-1, :]
        normal = normal / np.linalg.norm(normal)
        return normal, centroid

    def rotation_matrix_from_vectors(a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        if np.allclose(a, 0) or np.allclose(b, 0):
            raise ValueError('Zero vector provided')
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        v = np.cross(a, b)
        c = np.dot(a, b)
        if np.allclose(v, 0):
            if c > 0:
                return np.eye(3)
            else:
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

from matplotlib import pyplot as plt


def collect_points(reconstruction, downsample=1.0, rng_seed=42):
    pts = []
    for pid, p in reconstruction.points3D.items():
        pts.append(np.array(p.xyz, dtype=float))
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=float)
    pts = np.vstack(pts)
    if downsample > 0 and downsample < 1.0:
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(pts.shape[0], size=int(np.ceil(downsample * pts.shape[0])), replace=False)
        pts = pts[idx]
    return pts


def compute_alignment(cam_centers, up='z'):
    normal, centroid = fit_plane_svd(cam_centers)
    up_map = {'z': np.array([0.0, 0.0, 1.0]), 'y': np.array([0.0, 1.0, 0.0]), 'x': np.array([1.0, 0.0, 0.0])}
    target_up = up_map[up]
    R = rotation_matrix_from_vectors(normal, target_up)
    T = np.eye(4)
    T[:3, :3] = R
    return R, T, centroid


def select_xy_inliers(points_xy, keep_frac=0.95):
    """Return a boolean mask for the central keep_frac of points in XY."""
    pts = np.asarray(points_xy, dtype=float)
    n = pts.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=bool)
    if keep_frac >= 1.0 or n < 4:
        return np.ones((n,), dtype=bool)
    if keep_frac <= 0.0:
        return np.ones((n,), dtype=bool)

    keep_n = int(np.floor(keep_frac * n))
    keep_n = max(3, min(n, keep_n))

    center = np.median(pts, axis=0)
    d2 = np.sum((pts - center) ** 2, axis=1)
    order = np.argsort(d2)
    mask = np.zeros((n,), dtype=bool)
    mask[order[:keep_n]] = True
    return mask


def build_grid(points_xy, grid_res, pad_frac=0.05):
    # points_xy: Nx2
    xmin = float(np.min(points_xy[:, 0]))
    xmax = float(np.max(points_xy[:, 0]))
    ymin = float(np.min(points_xy[:, 1]))
    ymax = float(np.max(points_xy[:, 1]))
    dx = xmax - xmin
    dy = ymax - ymin
    if dx == 0:
        dx = 1e-3
        xmin -= dx / 2; xmax += dx / 2
    if dy == 0:
        dy = 1e-3
        ymin -= dy / 2; ymax += dy / 2
    xmin -= pad_frac * dx
    xmax += pad_frac * dx
    ymin -= pad_frac * dy
    ymax += pad_frac * dy
    width = int(np.ceil((xmax - xmin) / grid_res))
    height = int(np.ceil((ymax - ymin) / grid_res))
    # ensure at least 1 pixel
    width = max(1, width)
    height = max(1, height)
    x_edges = np.linspace(xmin, xmin + width * grid_res, width + 1)
    y_edges = np.linspace(ymin, ymin + height * grid_res, height + 1)
    meta = dict(xmin=xmin, xmax=xmin + width * grid_res, ymin=ymin, ymax=ymin + height * grid_res, width=width, height=height)
    return x_edges, y_edges, meta


def compute_density(points_rot, grid_res=0.02, height_bins=48, min_height=None, max_height=None,
                    ignore_low_frac=0.1, ignore_high_frac=0.1, xy_keep_frac=0.95):
    # points_rot: Nx3 (already rotated and optionally translated)
    if points_rot.shape[0] == 0:
        return None, None, None, None

    xy = points_rot[:, :2]
    z = points_rot[:, 2]

    # Keep only central XY footprint so isolated far outliers do not explode the grid size.
    inlier_mask = select_xy_inliers(xy, keep_frac=xy_keep_frac)
    if not np.any(inlier_mask):
        inlier_mask = np.ones((xy.shape[0],), dtype=bool)

    xy_used = xy[inlier_mask]
    z_used = z[inlier_mask]
    x_edges, y_edges, meta = build_grid(xy_used, grid_res)

    # Height bin edges
    if min_height is None:
        min_h = float(np.min(z_used))
    else:
        min_h = float(min_height)
    if max_height is None:
        max_h = float(np.max(z_used))
    else:
        max_h = float(max_height)
    if min_h >= max_h:
        # expand a tiny bit
        max_h = min_h + 1e-3
    h_edges = np.linspace(min_h, max_h, height_bins + 1)

    # 3D histogram (x, y, h)
    samples = np.vstack([xy_used[:, 0], xy_used[:, 1], z_used]).T
    H, edges = np.histogramdd(samples, bins=(x_edges, y_edges, h_edges))
    # H shape: (len(x_edges)-1, len(y_edges)-1, len(h_edges)-1)
    # we want x across cols and y across rows -> transpose later

    nbins_h = H.shape[2]
    ilow = int(np.floor(ignore_low_frac * nbins_h))
    ihigh = int(np.ceil(ignore_high_frac * nbins_h))
    start = ilow
    end = nbins_h - ihigh
    if start >= end:
        # fallback: pick middle half
        start = nbins_h // 4
        end = nbins_h - start
    selected = H[:, :, start:end]
    density2d = np.sum(selected, axis=2)
    meta['points_total'] = int(points_rot.shape[0])
    meta['points_kept_xy'] = int(np.count_nonzero(inlier_mask))
    meta['xy_keep_frac'] = float(xy_keep_frac)
    # density2d shape (nx, ny). For image we want (ny, nx) with origin lower-left; we'll transpose and flip as needed when saving
    return density2d, meta, dict(x_edges=x_edges, y_edges=y_edges, h_edges=h_edges), inlier_mask


def render_density_png(density2d, out_path, cmap='magma', normalize=True, vmax_percentile=99.5):
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    arr = density2d.copy()
    if normalize:
        vmax = np.percentile(arr[arr > 0], vmax_percentile) if np.any(arr > 0) else 1.0
        norm = Normalize(vmin=0.0, vmax=max(1.0, float(vmax)))
        cmap_inst = cm.get_cmap(cmap)
        rgba = cmap_inst(norm(arr))
        # convert to uint8 RGB
        rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    else:
        # scale linearly
        maxv = float(arr.max()) if arr.max() > 0 else 1.0
        rgb = (np.clip(arr / maxv, 0.0, 1.0) * 255).astype(np.uint8)
    # density2d currently indexed as (nx, ny) -> transpose to (ny, nx)
    rgb = np.transpose(rgb, (1, 0, 2))
    # flip vertically so that increasing Y goes up in image (origin lower-left)
    rgb = np.flipud(rgb)
    plt.imsave(str(out_path), rgb)


def render_grayscale_png(density2d, out_path):
    import matplotlib.pyplot as plt
    arr = density2d.copy()
    
    mask = arr > 0
    if np.any(mask):
        mean_val = np.mean(arr[mask])
        std_val = np.std(arr[mask])
        maxv = mean_val + std_val
        if maxv <= 0:
            maxv = 1.0
    else:
        maxv = 1.0

    # Linear scale using mean+std as max
    normed = arr / maxv
    normed = np.clip(normed, 0.0, 1.0)
    
    gray_vals = (255 - (normed * 255)).astype(np.uint8)
    
    # Ensure any non-zero bin is visibly dark (e.g. at least grey, not pure white)
    gray_vals[mask] = np.clip(gray_vals[mask], 0, 200)
    
    rgb = np.stack((gray_vals, gray_vals, gray_vals), axis=-1)
    rgb = np.transpose(rgb, (1, 0, 2))
    rgb = np.flipud(rgb)
    plt.imsave(str(out_path), rgb)


def overlay_cameras(density2d, meta, cams_rot, out_path, names=None, marker_color='red', marker_size=6):
    # density2d: (nx, ny)
    # meta: xmin,ymin,width,height,grid_res info
    width = meta['width']
    height = meta['height']
    xmin = meta['xmin']
    ymin = meta['ymin']
    grid_res = (meta['xmax'] - meta['xmin']) / float(width)
    # create image with density colormap (reuse render pipeline to construct RGB array)
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    arr = density2d.copy()
    vmax = np.percentile(arr[arr > 0], 99.5) if np.any(arr > 0) else 1.0
    norm = Normalize(vmin=0.0, vmax=max(1.0, float(vmax)))
    cmap_inst = cm.get_cmap('magma')
    rgba = cmap_inst(norm(arr))
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    rgb = np.transpose(rgb, (1, 0, 2))
    rgb = np.flipud(rgb)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb, origin='upper')
    # plot cameras: convert world XY to pixel
    xs = cams_rot[:, 0]
    ys = cams_rot[:, 1]
    # col = (x - xmin)/grid_res; row = height - 1 - (y - ymin)/grid_res (because image origin upper-left)
    cols = (xs - xmin) / grid_res
    rows = (meta['ymax'] - ys) / grid_res
    ax.scatter(cols, rows, c=marker_color, s=marker_size)
    if names is not None:
        for i, nm in enumerate(names):
            ax.text(cols[i] + 2, rows[i] + 2, nm, color='white', fontsize=6)
    ax.set_axis_off()
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)


def save_projected_csv(points_rot, meta, out_csv_path):
    # Save x, y, z and pixel coords
    xmin = meta['xmin']
    ymin = meta['ymin']
    grid_res = (meta['xmax'] - meta['xmin']) / float(meta['width'])
    with open(out_csv_path, 'w', newline='') as cf:
        writer = csv.writer(cf)
        writer.writerow(['x', 'y', 'z', 'x_px', 'y_px'])
        for p in points_rot:
            x, y, z = float(p[0]), float(p[1]), float(p[2])
            x_px = (x - xmin) / grid_res
            y_px = (meta['ymax'] - y) / grid_res
            writer.writerow([x, y, z, x_px, y_px])


def show_stage_visualizations(density2d, meta, bins, pts_rot, cams_before, cams_rot, centers_info, out_dir):
    """Create a multi-panel figure showing intermediate results and save it.

    Panels:
    - Top-down sieved density (as in density.png)
    - Camera XY before and after rotation
    - Z histogram with selected height range highlighted
    - A small image of the summed selected height-slices
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    # Prepare density RGB for display
    arr = density2d.copy()
    vmax = np.percentile(arr[arr > 0], 99.5) if np.any(arr > 0) else 1.0
    norm = Normalize(vmin=0.0, vmax=max(1.0, float(vmax)))
    cmap_inst = cm.get_cmap('magma')
    rgba = cmap_inst(norm(arr))
    rgb_img = (rgba[:, :, :3] * 255).astype(np.uint8)
    rgb_img = np.transpose(rgb_img, (1, 0, 2))
    rgb_img = np.flipud(rgb_img)

    # Z histogram
    zs = pts_rot[:, 2]
    h_edges = bins['h_edges']
    nb = len(h_edges) - 1
    counts_per_bin = np.histogram(zs, bins=h_edges)[0]
    ilow = int(np.floor(0.0 * nb))
    ihigh = int(np.ceil(0.0 * nb))

    fig = plt.figure(figsize=(12, 9))
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((3, 3), (0, 2))
    ax3 = plt.subplot2grid((3, 3), (1, 0), colspan=1)
    ax4 = plt.subplot2grid((3, 3), (1, 1), colspan=2)

    # Top-down density
    ax1.imshow(rgb_img)
    ax1.set_title('Sieved density (top-down)')
    ax1.axis('off')

    # Camera centers before/after (XY)
    ax2.set_title('Cameras XY before (blue) / after (red)')
    if cams_before is not None and cams_before.shape[0] > 0:
        ax2.scatter(cams_before[:, 0], cams_before[:, 1], c='blue', s=8, label='before')
    ax2.scatter(cams_rot[:, 0], cams_rot[:, 1], c='red', s=12, label='after')
    ax2.legend()
    ax2.set_xlabel('X'); ax2.set_ylabel('Y')

    # Z histogram
    ax3.bar((h_edges[:-1] + h_edges[1:]) / 2.0, counts_per_bin, width=(h_edges[1] - h_edges[0]))
    ax3.set_title('Z histogram (points)')
    ax3.set_xlabel('Z'); ax3.set_ylabel('counts')

    # small view of summed selected slices (density2d inverted axes handled already)
    ax4.imshow(rgb_img)
    if centers_info is not None:
        names = [n for (_, n, _) in centers_info]
        # project camera pixels to overlay
        xmin = meta['xmin']; grid_res = (meta['xmax'] - meta['xmin']) / float(meta['width'])
        cols = (cams_rot[:, 0] - xmin) / grid_res
        rows = (meta['ymax'] - cams_rot[:, 1]) / grid_res
        ax4.scatter(cols, rows, c='lime', s=10)
    ax4.set_title('Overlay: cameras (lime)')
    ax4.axis('off')

    plt.tight_layout()
    out_png = pathlib.Path(out_dir) / 'sieved_stages.png'
    fig.savefig(str(out_png), dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description='Project COLMAP scene to top-down plan and highlight vertical structures')
    p.add_argument('reconstruction', type=pathlib.Path, help='Path to COLMAP reconstruction folder (cameras.bin, images.bin, points3D.bin)')
    p.add_argument('--output_dir', type=pathlib.Path, default=None, help='Directory to save outputs (default: reconstruction parent/view_or)')
    p.add_argument('--up', choices=['z', 'y', 'x'], default='z', help='Which axis is up after orientation (default z)')
    p.add_argument('--grid_res', type=float, default=0.02, help='Grid resolution in meters/pixel')
    p.add_argument('--height_bins', type=int, default=48, help='Number of height bins to slice vertically')
    p.add_argument('--min_height', type=float, default=None, help='Minimum height (z) to consider for bins (default: auto)')
    p.add_argument('--max_height', type=float, default=None, help='Maximum height (z) to consider for bins (default: auto)')
    p.add_argument('--ignore_low_frac', type=float, default=0.1, help='Fraction of lowest height bins to ignore (floor)')
    p.add_argument('--ignore_high_frac', type=float, default=0.1, help='Fraction of highest height bins to ignore (ceiling)')
    p.add_argument('--xy_keep_frac', type=float, default=0.95, help='Fraction of central XY points kept for grid/projection bounds')
    p.add_argument('--downsample', type=float, default=1.0, help='Fraction of points to sample (0-1)')
    p.add_argument('--set_ground', action='store_true', help='Translate so minimum Z after rotation becomes 0')
    p.add_argument('--save_ply', action='store_true', help='Save rotated points as PLY (ASCII)')
    p.add_argument('--show', action='store_true', help='Show images interactively')
    p.add_argument('--show_stages', action='store_true', help='Show intermediate stages of processing')
    args = p.parse_args()

    rec_path = args.reconstruction
    if not rec_path.exists():
        print('Reconstruction path does not exist:', rec_path)
        sys.exit(2)

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = rec_path.parent / 'view_or'
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('Loading reconstruction from', rec_path)
    reconstruction = pycolmap.Reconstruction(str(rec_path))

    centers_info = get_camera_centers(reconstruction)
    if len(centers_info) < 3:
        print('Need at least 3 camera centers to estimate plane. Found:', len(centers_info))
        sys.exit(3)
    cam_centers = np.vstack([c for (_, _, c) in centers_info])

    pts3 = collect_points(reconstruction, downsample=args.downsample)
    if pts3.shape[0] == 0:
        print('No 3D points found in reconstruction.')
        sys.exit(4)

    R, T, centroid = compute_alignment(cam_centers, up=args.up)
    pts_rot = (R.dot(pts3.T)).T
    cams_rot = (R.dot(cam_centers.T)).T

    # Optionally translate so min Z = 0
    trans = np.zeros(3)
    if args.set_ground:
        all_z = np.concatenate([pts_rot[:, 2], cams_rot[:, 2]])
        min_z = np.min(all_z)
        trans[2] = -min_z
        pts_rot = pts_rot + trans
        cams_rot = cams_rot + trans
        T[:3, 3] = trans

    # Save transform
    tf_path = out_dir / 'transform.txt'
    np.savetxt(tf_path, T, fmt='%.6f')
    print('Saved transform to', tf_path)

    # Compute density
    density2d, meta, bins, inlier_mask = compute_density(
        pts_rot,
        grid_res=args.grid_res,
        height_bins=args.height_bins,
        min_height=args.min_height,
        max_height=args.max_height,
        ignore_low_frac=args.ignore_low_frac,
        ignore_high_frac=args.ignore_high_frac,
        xy_keep_frac=args.xy_keep_frac,
    )
    if density2d is None:
        print('No density produced (empty points).')
        sys.exit(5)

    print(f"Using {meta['points_kept_xy']} / {meta['points_total']} points for XY footprint (xy_keep_frac={meta['xy_keep_frac']:.3f})")

    density_png = out_dir / 'density.png'
    render_density_png(density2d, density_png)
    print('Saved density image to', density_png)

    # also save a sieved view for quick inspection
    sieved_png = out_dir / 'sieved.png'
    render_density_png(density2d, sieved_png)
    print('Saved sieved image to', sieved_png)

    gray_png = out_dir / 'density_bw.png'
    render_grayscale_png(density2d, gray_png)
    print('Saved grayscale image to', gray_png)

    overlay_png = out_dir / 'density_overlay.png'
    overlay_cameras(density2d, meta, cams_rot, overlay_png, names=[n for (_, n, _) in centers_info])
    print('Saved overlay image to', overlay_png)

    csv_path = out_dir / 'projected_points.csv'
    save_projected_csv(pts_rot[inlier_mask], meta, csv_path)
    print('Saved projected points CSV to', csv_path)

    if args.save_ply:
        ply_path = out_dir / 'rotated_points.ply'
        write_ply_ascii(pts_rot, ply_path)
        print('Saved rotated points PLY to', ply_path)

    if args.show:
        # show overlay image and sieved stages (create stages and show them side-by-side)
        cams_before = cam_centers
        show_stage_visualizations(density2d, meta, bins, pts_rot[inlier_mask], cams_before, cams_rot, centers_info, out_dir)
        stages_png = out_dir / 'sieved_stages.png'

        img_overlay = plt.imread(str(overlay_png))
        img_stages = plt.imread(str(stages_png))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        ax1.imshow(img_overlay)
        ax1.set_title('Density overlay')
        ax1.axis('off')
        ax2.imshow(img_stages)
        ax2.set_title('Sieved stages')
        ax2.axis('off')
        plt.tight_layout()
        plt.show()

    # new: show stages interactive multi-panel (optional)
    if hasattr(args, 'show_stages') and args.show_stages:
        # Keep original camera centers before rotation for comparison
        cams_before = cam_centers
        show_stage_visualizations(density2d, meta, bins, pts_rot[inlier_mask], cams_before, cams_rot, centers_info, out_dir)
        print('Saved stage visualization to', out_dir / 'sieved_stages.png')


if __name__ == '__main__':
    main()
