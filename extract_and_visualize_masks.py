#!/usr/bin/env python3
"""
Extract point-map points and separate walls and floor regions from a floor-plan image.

This script does NOT perform any alignment. It reads two images (point_map and floor_map),
extracts:
 - point mask from the point_map (dark pixels)
 - wall mask from the floor_map (almost-black pixels)
 - floor mask from the floor_map (green corridor detection in HSV)

It then displays a single overlay window showing the floor plan with colored highlights:
 - walls = red
 - floor (green corridor) = blue
 - point-map points = yellow

Usage:
    python extract_and_visualize_masks.py --point <point_map> --floor <floor_map> [--save PREFIX]

Defaults use the repository test files.
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
try:
    from scipy.spatial import cKDTree as KDTree
except Exception:
    from scipy.spatial import KDTree as KDTree


def minmax_norm(img):
    f = img.astype(np.float32)
    mi = f.min()
    ma = f.max()
    if ma - mi <= 1e-6:
        return np.zeros_like(f)
    return (f - mi) / (ma - mi)


def extract_point_mask(point_path, thresh=0.5, max_points=None, nonwhite=False, white_thresh=0.99):
    img = cv2.imread(str(point_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f'Failed to load point image: {point_path}')
    norm = minmax_norm(img)
    if nonwhite:
        # keep all non-white pixels; white_thresh is in normalized [0,1]
        mask = (norm < float(white_thresh)).astype(np.uint8)
    else:
        mask = (norm < float(thresh)).astype(np.uint8)
    if max_points is not None and max_points > 0:
        pts = np.column_stack(np.nonzero(mask))[:, ::-1]
        if pts.shape[0] > max_points:
            idx = np.random.choice(pts.shape[0], max_points, replace=False)
            keep = np.zeros(mask.shape, dtype=np.uint8)
            sel = pts[idx]
            keep[sel[:,1], sel[:,0]] = 1
            mask = keep
    return img, mask


def extract_floor_masks(floor_path, wall_thresh=0.15, morph_k=3):
    img = cv2.imread(str(floor_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f'Failed to load floor image: {floor_path}')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    norm = minmax_norm(gray)
    wall_mask = (norm < float(wall_thresh)).astype(np.uint8)
    # clean
    if morph_k > 0:
        K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
        wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_OPEN, K)
        wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, K)

    # floor (green) detection in HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # tunable green range
    lower = np.array([35, 40, 40])
    upper = np.array([95, 255, 255])
    floor_mask = cv2.inRange(hsv, lower, upper) // 255
    if morph_k > 0:
        K = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_k, morph_k))
        floor_mask = cv2.morphologyEx(floor_mask.astype(np.uint8), cv2.MORPH_OPEN, K)
    return img, wall_mask.astype(np.uint8), floor_mask.astype(np.uint8)


def _scale_for_display(img, max_dim=1280):
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = max_dim / float(max(h, w))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def build_overlay(floor_img, wall_mask, floor_mask, point_mask, white_bg=False, draw_floor=False, point_color=(255, 0, 0)):
    """Build an overlay image.

    Args:
        floor_img: background image (grayscale or BGR)
        wall_mask: binary mask for walls
        floor_mask: binary mask for corridor/floor
        point_mask: binary mask for point cloud
        white_bg: if True, use a white background instead of floor_img
        draw_floor: if True, draw the floor_mask (disabled by default)
        point_color: BGR color tuple for point cloud (default blue)
    """
    if white_bg:
        h, w = floor_img.shape[:2]
        bg = np.full((h, w, 3), 255, dtype=np.uint8)
        overlay = bg.copy()
        # walls red
        overlay[wall_mask > 0] = [0, 0, 255]
        # optionally paint floor
        if draw_floor:
            overlay[floor_mask > 0] = [255, 0, 0]
        # paint points with provided color
        overlay[point_mask > 0] = list(point_color)
        return overlay
    else:
        if len(floor_img.shape) == 2:
            bg = cv2.cvtColor(floor_img, cv2.COLOR_GRAY2BGR)
        else:
            bg = floor_img.copy()
        overlay = bg.copy()
        # walls red
        overlay[wall_mask > 0] = [0, 0, 255]
        if draw_floor:
            overlay[floor_mask > 0] = [255, 0, 0]
        overlay[point_mask > 0] = list(point_color)
        blended = cv2.addWeighted(bg, 0.6, overlay, 0.4, 0)
        return blended


def _points_from_mask(mask):
    # returns Nx2 array of float (x,y)
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    pts = np.column_stack((xs, ys)).astype(np.float64)
    return pts


def umeyama_similarity(src, dst, with_scale=True):
    # src,dst: (N,2) numpy arrays
    assert src.shape == dst.shape and src.shape[0] >= 1
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    cov = (dst_c.T @ src_c) / src.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    if with_scale:
        var_src = (src_c ** 2).sum() / src.shape[0]
        if var_src <= 0:
            s = 1.0
        else:
            s = float(np.trace(np.diag(D) @ S) / var_src)
    else:
        s = 1.0
    t = dst_mean - s * (R @ src_mean)
    return s, R, t


def icp_similarity(src_pts, tgt_pts, max_iters=50, tol=1e-6, with_scale=True, max_sample=5000):
    # Optionally subsample points for speed
    if src_pts.shape[0] > max_sample:
        idx = np.random.choice(src_pts.shape[0], max_sample, replace=False)
        src = src_pts[idx].copy()
    else:
        src = src_pts.copy()
    if tgt_pts.shape[0] > max_sample:
        idx2 = np.random.choice(tgt_pts.shape[0], max_sample, replace=False)
        tgt = tgt_pts[idx2].copy()
    else:
        tgt = tgt_pts.copy()

    tree = KDTree(tgt)
    cur_src = src.copy()
    prev_error = float('inf')
    s_total = 1.0
    R_total = np.eye(2)
    t_total = np.zeros(2)
    for it in range(max_iters):
        dists, idx = tree.query(cur_src)
        matched = tgt[idx]
        s, R, t = umeyama_similarity(cur_src, matched, with_scale=with_scale)
        # apply to current source
        cur_src = (s * (cur_src @ R.T)) + t
        mean_err = float(np.mean(dists**2))
        # accumulate transforms
        s_total_new = s * s_total
        R_total_new = R @ R_total
        t_total_new = s * (R @ t_total) + t
        s_total, R_total, t_total = s_total_new, R_total_new, t_total_new
        if abs(prev_error - mean_err) < tol:
            break
        prev_error = mean_err
    return s_total, R_total, t_total, prev_error


def affine_from_sim(s, R, t):
    M = np.zeros((2, 3), dtype=np.float32)
    M[0, 0] = s * R[0, 0]
    M[0, 1] = s * R[0, 1]
    M[0, 2] = t[0]
    M[1, 0] = s * R[1, 0]
    M[1, 1] = s * R[1, 1]
    M[1, 2] = t[1]
    return M


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--point', '-p', type=str, default='colmap_projects/floor_1/png/density_bw.png')
    parser.add_argument('--floor', '-f', type=str, default='plan/20260428_112650_crop.jpg')
    parser.add_argument('--wall-thresh', type=float, default=0.15, help='Normalized threshold for walls (floor plan)')
    parser.add_argument('--point-thresh', type=float, default=0.5, help='Normalized threshold for point map')
    parser.add_argument('--save', type=str, default=None, help='Prefix to save masks and overlay')
    parser.add_argument('--white-bg', dest='white_bg', action='store_true', help='Make non-colored pixels white in overlay')
    parser.add_argument('--point-nonwhite', dest='point_nonwhite', action='store_true', help='Treat all non-white pixels in point map as points (ignore dark-only threshold)')
    parser.add_argument('--point-white-thresh', dest='point_white_thresh', type=float, default=0.99, help='Normalized threshold to consider a pixel white for point-nonwhite mode')
    parser.add_argument('--align-icp', dest='align_icp', action='store_true', help='Run ICP to align point cloud to wall points')
    parser.add_argument('--icp-iters', dest='icp_iters', type=int, default=50, help='ICP max iterations')
    parser.add_argument('--icp-no-scale', dest='icp_no_scale', action='store_true', help="Don't estimate scale during ICP (rigid only)")
    parser.add_argument('--icp-max-sample', dest='icp_max_sample', type=int, default=5000, help='Max points for ICP subsampling')
    args = parser.parse_args()

    prefix = args.save

    point_path = Path(args.point)
    floor_path = Path(args.floor)
    if not point_path.exists() or not floor_path.exists():
        print('Files not found:', point_path, floor_path)
        return

    point_img, point_mask = extract_point_mask(point_path, thresh=args.point_thresh, nonwhite=args.point_nonwhite, white_thresh=args.point_white_thresh)
    floor_img, wall_mask, floor_mask = extract_floor_masks(floor_path, wall_thresh=args.wall_thresh)

    # If point map and floor map have different sizes, resize point mask to floor image size
    h_f, w_f = floor_img.shape[:2]
    def _resize_preserve_aspect(src_mask, target_shape):
        h_t, w_t = target_shape
        h_s, w_s = src_mask.shape[:2]
        if (h_s, w_s) == (h_t, w_t):
            return src_mask
        # compute scale to fit while preserving aspect
        scale = min(w_t / float(w_s), h_t / float(h_s))
        new_w = max(1, int(round(w_s * scale)))
        new_h = max(1, int(round(h_s * scale)))
        resized = cv2.resize(src_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        # place centered on target canvas
        canvas = np.zeros((h_t, w_t), dtype=resized.dtype)
        x_off = (w_t - new_w) // 2
        y_off = (h_t - new_h) // 2
        canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
        return canvas

    if point_mask.shape[:2] != (h_f, w_f):
        point_mask_resized = _resize_preserve_aspect(point_mask, (h_f, w_f))
    else:
        point_mask_resized = point_mask

    overlay = build_overlay(floor_img, wall_mask, floor_mask, point_mask_resized, white_bg=args.white_bg, draw_floor=False, point_color=(255,0,0))
    disp = _scale_for_display(overlay, max_dim=1280)

    win = 'extract_overlay'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.imshow(win, disp)
    print('Showing overlay. Press any key in the window to exit.')
    if args.save:
        prefix = args.save
        cv2.imwrite(f"{prefix}_overlay.png", overlay)
        cv2.imwrite(f"{prefix}_wall_mask.png", (wall_mask * 255).astype(np.uint8))
        cv2.imwrite(f"{prefix}_floor_mask.png", (floor_mask * 255).astype(np.uint8))
        cv2.imwrite(f"{prefix}_point_mask.png", (point_mask_resized * 255).astype(np.uint8))
        print('Saved masks and overlay with prefix', prefix)

    # Optional ICP alignment: align point mask points to wall mask points
    if args.align_icp:
        print('Running ICP to align point cloud to wall points...')
        src_pts = _points_from_mask(point_mask_resized)
        tgt_pts = _points_from_mask(wall_mask)
        if src_pts.shape[0] == 0 or tgt_pts.shape[0] == 0:
            print('No source or target points for ICP; skipping')
        else:
            s_tot, R_tot, t_tot, err = icp_similarity(src_pts, tgt_pts, max_iters=args.icp_iters,
                                                      with_scale=not args.icp_no_scale, max_sample=args.icp_max_sample)
            M = affine_from_sim(s_tot, R_tot, t_tot)
            print(f'ICP result: scale={s_tot:.6f}, rotation_deg={np.degrees(np.arctan2(R_tot[1,0], R_tot[0,0])):.3f}, tx={t_tot[0]:.2f}, ty={t_tot[1]:.2f}, err={err:.6f}')
            # warp original point image for visualization
            warped_points_img = cv2.warpAffine(point_img, M, (floor_img.shape[1], floor_img.shape[0]), flags=cv2.INTER_LINEAR, borderValue=255)
            # build overlay showing warped points on white or floor
            if args.white_bg:
                final_overlay = build_overlay(floor_img, wall_mask, floor_mask, (warped_points_img < 200).astype(np.uint8), white_bg=True, draw_floor=False, point_color=(255,0,0))
            else:
                final_overlay = build_overlay(floor_img, wall_mask, floor_mask, (warped_points_img < 200).astype(np.uint8), white_bg=False, draw_floor=False, point_color=(255,0,0))
            disp = _scale_for_display(final_overlay, max_dim=1280)
            cv2.imshow(win, disp)
            if args.save:
                cv2.imwrite(f"{prefix}_overlay_icp.png", final_overlay)
                # save warp matrix
                np.savetxt(f"{prefix}_icp_matrix.txt", M)

    cv2.waitKey(0)
    try:
        cv2.destroyWindow(win)
    except Exception:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()


