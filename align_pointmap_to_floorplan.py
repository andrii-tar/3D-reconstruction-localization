#!/usr/bin/env python3
"""
Align a point-cloud projection image (point_map) onto a floor plan image (floor_map).

Allows rotation, isotropic scaling, translation and optional single horizontal/vertical flip.
Saves an overlay image and prints the best transform parameters.

Usage:
    python align_pointmap_to_floorplan.py \
        --point D:/.../density_bw.png \
        --floor D:/.../20260428_112650_crop.jpg

Defaults use the files in the workspace for quick testing.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
try:
    from scipy.spatial import cKDTree as KDTree
except Exception:
    from scipy.spatial import KDTree as KDTree
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from types import SimpleNamespace


def extract_points_from_image(path, invert=False, max_points=20000, thresh=None):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    # Min-max normalization to [0,1]
    img_f = img.astype(np.float32)
    mi = float(img_f.min())
    ma = float(img_f.max())
    if ma - mi <= 1e-6:
        norm = np.zeros_like(img_f)
    else:
        norm = (img_f - mi) / (ma - mi)
    # Threshold normalized image to obtain binary mask (walls/points are dark -> low values)
    if thresh is None:
        thresh = 0.5
    bin_img = (norm < float(thresh)).astype(np.uint8)
    if invert:
        bin_img = 1 - bin_img
    # pts are (x,y) coordinates where mask is True
    pts = np.column_stack(np.nonzero(bin_img))[:, ::-1].astype(np.float64)
    # pts are (x, y) coordinates
    if pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], max_points, replace=False)
        pts = pts[idx]
    return pts, img, bin_img


def build_affine_matrix(s, theta, tx, ty, flip_x, flip_y):
    c = np.cos(theta)
    si = np.sin(theta)
    F = np.array([[ -1.0 if flip_x else 1.0, 0.0], [0.0, -1.0 if flip_y else 1.0]])
    R = np.array([[c, -si], [si, c]])
    A2 = s * R.dot(F)
    M = np.array([[A2[0,0], A2[0,1], tx], [A2[1,0], A2[1,1], ty]], dtype=np.float32)
    return M


def transform_points(pts, s, theta, tx, ty, flip_x=False, flip_y=False):
    F = np.array([[-1.0 if flip_x else 1.0, -0.0], [-0.0, -1.0 if flip_y else 1.0]])
    ptsF = pts * np.array([1.0, 1.0])
    if flip_x or flip_y:
        ptsF = ptsF.copy()
        ptsF[:, 0] *= (-1.0 if flip_x else 1.0)
        ptsF[:, 1] *= (-1.0 if flip_y else 1.0)
    c = np.cos(theta)
    si = np.sin(theta)
    R = np.array([[c, -si], [si, c]])
    out = s * (ptsF.dot(R.T)) + np.array([tx, ty])
    return out


def objective_xy(params, src_pts, tgt_kdtree, flip_x, flip_y):
    # parameterization: params = [log_s, theta, tx, ty] to enforce s>0
    log_s, theta, tx, ty = params
    s = float(np.exp(log_s))
    trans = transform_points(src_pts, s, theta, tx, ty, flip_x, flip_y)
    # cKDTree in some SciPy versions supports n_jobs; older KDTree does not.
    # Call without n_jobs for compatibility and handle both return signatures.
    try:
        q = tgt_kdtree.query(trans, k=1)
    except TypeError:
        # Older implementations may raise TypeError for unexpected args; fallback
        q = tgt_kdtree.query(trans)
    if isinstance(q, tuple):
        dists = q[0]
    else:
        dists = q
    # Use mean squared distance
    return float(np.mean(dists**2))


def fit_similarity(src_pts, tgt_pts, flip_x=False, flip_y=False, max_sample=3000):
    # Use a simple ICP-like loop with closed-form Umeyama similarity estimation.
    if len(src_pts) == 0 or len(tgt_pts) == 0:
        raise RuntimeError('Empty point sets for fitting')

    def umeyama_sim(src, dst, with_scale=True):
        # src, dst: (N,2)
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
            s = float(np.trace(np.diag(D) @ S) / var_src)
        else:
            s = 1.0
        t = dst_mean - s * (R @ src_mean)
        return s, R, t

    # Subsample for speed; if max_sample <= 0 use all points
    if max_sample is None or max_sample <= 0:
        s_idx = np.arange(src_pts.shape[0])
        t_idx = np.arange(tgt_pts.shape[0])
    else:
        if src_pts.shape[0] > max_sample:
            s_idx = np.random.choice(src_pts.shape[0], max_sample, replace=False)
        else:
            s_idx = np.arange(src_pts.shape[0])
        if tgt_pts.shape[0] > max_sample:
            t_idx = np.random.choice(tgt_pts.shape[0], max_sample, replace=False)
        else:
            t_idx = np.arange(tgt_pts.shape[0])

    src_samp = src_pts[s_idx]
    tgt_samp = tgt_pts[t_idx]
    # apply flip to src_samp before ICP
    if flip_x:
        src_samp = src_samp.copy(); src_samp[:, 0] *= -1
    if flip_y:
        src_samp = src_samp.copy(); src_samp[:, 1] *= -1

    tgt_tree = KDTree(tgt_samp)
    prev_err = float('inf')
    cur_src = src_samp.copy()
    best = SimpleNamespace(fun=float('inf'), x=None)
    for it in range(40):
        dists, idx = tgt_tree.query(cur_src, k=1)
        matched = tgt_samp[idx]
        s, R, t = umeyama_sim(cur_src, matched, with_scale=True)
        # apply transform to original sampled src for next iteration
        cur_src = (s * (cur_src @ R.T)) + t
        err = float(np.mean(dists ** 2))
        if err < best.fun:
            # store parameters as log_s, theta, tx, ty
            theta = float(np.arctan2(R[1, 0], R[0, 0]))
            log_s = float(np.log(s))
            tx, ty = float(t[0]), float(t[1])
            best.fun = err
            best.x = np.array([log_s, theta, tx, ty])
        if abs(prev_err - err) < 1e-6:
            break
        prev_err = err

    return best


def overlay_and_save(src_img, floor_img, M, out_path):
    h, w = floor_img.shape[:2]
    # warp src into target frame
    warped = cv2.warpAffine(src_img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
    # create color overlay: floor as grayscale background, warped points in red
    if len(floor_img.shape) == 2:
        floor_color = cv2.cvtColor(floor_img, cv2.COLOR_GRAY2BGR)
    else:
        floor_color = floor_img.copy()
    # Where warped is dark (points), color red
    mask = warped < 200
    overlay = floor_color.copy()
    overlay[mask] = [0, 0, 255]
    cv2.imwrite(str(out_path), overlay)
    return overlay


def _build_overlay(tgt_img, warped_mask):
    # tgt_img: grayscale or BGR image, warped_mask: uint8 mask (0/255)
    if tgt_img is None:
        # produce a gray background
        h, w = warped_mask.shape[:2]
        bg = np.full((h, w, 3), 200, dtype=np.uint8)
    else:
        if len(tgt_img.shape) == 2:
            bg = cv2.cvtColor(tgt_img, cv2.COLOR_GRAY2BGR)
        else:
            bg = tgt_img.copy()
    mask = (warped_mask > 0)
    overlay = bg.copy()
    overlay[mask] = [0, 0, 255]
    # blend
    blended = cv2.addWeighted(bg, 0.6, overlay, 0.4, 0)
    return blended


def _scale_for_display(img, max_dim=1280):
    """Scale image so the largest side equals max_dim, keep aspect ratio. Return uint8 image."""
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = max_dim / float(max(h, w))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized


def align_masks_by_center(src_bin, tgt_bin, tgt_img=None, try_flips=True, coarse_step=5, refine=True,
                          visualize=False, save_steps=False, out_prefix='step', center_align=True):
    """Align binary masks using center-of-mass, scale by bounding boxes, and rotate to minimize distance transform loss.

    src_bin, tgt_bin are uint8 single-channel masks with foreground 1 (or 255) and background 0.
    Returns (best_M, best_params, best_loss) where best_params = dict(scale, angle_deg, tx, ty, flip_x, flip_y)
    """
    # ensure binary 0/1
    src = (src_bin > 0).astype(np.uint8)
    tgt = (tgt_bin > 0).astype(np.uint8)
    h, w = tgt.shape[:2]

    # prepare distance transform for target: distance to nearest foreground
    tgt_u8 = (tgt * 255).astype(np.uint8)
    inv = 255 - tgt_u8
    dt = cv2.distanceTransform(inv, cv2.DIST_L2, 3)

    flips = [(False, False)] if not try_flips else [(False, False), (True, False), (False, True), (True, True)]

    best = None
    step_i = 0
    window_name = 'alignment_preview'
    if visualize:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    for flip_x, flip_y in flips:
        # apply flips to source image
        s_img = src.copy()
        if flip_x:
            s_img = np.flip(s_img, axis=1).copy()
        if flip_y:
            s_img = np.flip(s_img, axis=0).copy()

        # compute centers of mass
        ys, xs = np.nonzero(s_img)
        if len(xs) == 0:
            continue
        src_cx = float(xs.mean())
        src_cy = float(ys.mean())
        ys2, xs2 = np.nonzero(tgt)
        if len(xs2) == 0:
            continue
        tgt_cx = float(xs2.mean())
        tgt_cy = float(ys2.mean())

        # compute bounding boxes sizes
        src_xmin, src_xmax = xs.min(), xs.max()
        src_ymin, src_ymax = ys.min(), ys.max()
        tgt_xmin, tgt_xmax = xs2.min(), xs2.max()
        tgt_ymin, tgt_ymax = ys2.min(), ys2.max()
        src_w = float(src_xmax - src_xmin + 1)
        src_h = float(src_ymax - src_ymin + 1)
        tgt_w = float(tgt_xmax - tgt_xmin + 1)
        tgt_h = float(tgt_ymax - tgt_ymin + 1)

        if src_w <= 0 or src_h <= 0:
            continue

        # scale to fit
        scale = min(tgt_w / src_w, tgt_h / src_h)
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0

        # compute translation so that scaled src center maps to tgt center (optional)
        if center_align:
            tx = tgt_cx - scale * src_cx
            ty = tgt_cy - scale * src_cy
        else:
            tx = 0.0
            ty = 0.0

        # prepare affine M0: scale about origin and translate (if center_align)
        M0 = np.array([[scale, 0.0, tx], [0.0, scale, ty]], dtype=np.float32)
        # warp source mask into target frame
        warped0 = cv2.warpAffine((s_img * 255).astype(np.uint8), M0, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)

        if visualize:
            # show initial center+scale result in the same window
            disp = _build_overlay(tgt_img, warped0)
            disp_show = _scale_for_display(disp, max_dim=1280)
            cv2.imshow(window_name, disp_show)
            if save_steps:
                cv2.imwrite(f"{out_prefix}_{step_i:03d}_init.png", disp)
            # wait until key pressed to continue
            key = cv2.waitKey(0) & 0xFF
            # if ESC or 'q' pressed, abort visualization early
            if key in (27, ord('q')):
                visualize = False
            step_i += 1

        # coarse rotation sweep
        angles = list(range(-180, 180, coarse_step))
        best_local = None
        for a in angles:
            R = cv2.getRotationMatrix2D((tgt_cx, tgt_cy), a, 1.0)
            warped = cv2.warpAffine(warped0, R, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
            # if center_align disabled, perform translation search via template matching
            if not center_align:
                # extract bbox of current warped foreground as template
                ys_s, xs_s = np.nonzero(warped > 0)
                if len(xs_s) == 0:
                    loss = float('inf')
                    shifted = warped
                else:
                    xmin, xmax = xs_s.min(), xs_s.max()
                    ymin, ymax = ys_s.min(), ys_s.max()
                    tpl = warped[ymin:ymax+1, xmin:xmax+1]
                    # match template in target mask (tgt_u8)
                    try:
                        res = cv2.matchTemplate(tgt_u8, tpl, cv2.TM_CCORR_NORMED)
                        _, _, _, max_loc = cv2.minMaxLoc(res)
                        tx_tl, ty_tl = max_loc
                        # build shifted warp
                        shifted = np.zeros_like(warped)
                        h_t, w_t = tpl.shape[:2]
                        # clip to image bounds
                        x0 = int(tx_tl)
                        y0 = int(ty_tl)
                        x1 = min(w, x0 + w_t)
                        y1 = min(h, y0 + h_t)
                        sx1 = x1 - x0
                        sy1 = y1 - y0
                        if sx1 > 0 and sy1 > 0:
                            shifted[y0:y1, x0:x1] = tpl[0:sy1, 0:sx1]
                        src_fg = (shifted > 0)
                        if src_fg.sum() == 0:
                            loss = float('inf')
                        else:
                            ys_s2, xs_s2 = np.nonzero(src_fg)
                            loss = float(np.mean(dt[ys_s2, xs_s2] ** 2))
                    except Exception:
                        loss = float('inf')
                        shifted = warped
            else:
                src_fg = (warped > 0)
                if src_fg.sum() == 0:
                    loss = float('inf')
                    shifted = warped
                else:
                    ys_s, xs_s = np.nonzero(src_fg)
                    loss = float(np.mean(dt[ys_s, xs_s] ** 2))
                    shifted = warped
            if best_local is None or loss < best_local[0]:
                best_local = (loss, a, shifted)
            if visualize:
                disp = _build_overlay(tgt_img, warped)
                disp_show = _scale_for_display(disp, max_dim=1280)
                cv2.imshow(window_name, disp_show)
                if save_steps:
                    cv2.imwrite(f"{out_prefix}_{step_i:03d}_rot_{a:03d}.png", disp)
                # short pause to allow quick browsing; if key pressed, stop visualization
                key = cv2.waitKey(50) & 0xFF
                if key in (27, ord('q')):
                    visualize = False
                step_i += 1

        if best_local is None:
            continue

        # optional refinement around best angle
        best_loss, best_angle, best_warp = best_local
        if refine:
            # refine in +/- coarse_step range with 1 degree steps
            low = best_angle - coarse_step
            high = best_angle + coarse_step + 1
            for a in range(low, high, 1):
                R = cv2.getRotationMatrix2D((tgt_cx, tgt_cy), a, 1.0)
                warped = cv2.warpAffine(warped0, R, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
                if not center_align:
                    ys_s, xs_s = np.nonzero(warped > 0)
                    if len(xs_s) == 0:
                        loss = float('inf')
                        shifted = warped
                    else:
                        xmin, xmax = xs_s.min(), xs_s.max()
                        ymin, ymax = ys_s.min(), ys_s.max()
                        tpl = warped[ymin:ymax+1, xmin:xmax+1]
                        try:
                            res = cv2.matchTemplate(tgt_u8, tpl, cv2.TM_CCORR_NORMED)
                            _, _, _, max_loc = cv2.minMaxLoc(res)
                            tx_tl, ty_tl = max_loc
                            shifted = np.zeros_like(warped)
                            h_t, w_t = tpl.shape[:2]
                            x0 = int(tx_tl)
                            y0 = int(ty_tl)
                            x1 = min(w, x0 + w_t)
                            y1 = min(h, y0 + h_t)
                            sx1 = x1 - x0
                            sy1 = y1 - y0
                            if sx1 > 0 and sy1 > 0:
                                shifted[y0:y1, x0:x1] = tpl[0:sy1, 0:sx1]
                            src_fg2 = (shifted > 0)
                            if src_fg2.sum() == 0:
                                loss = float('inf')
                            else:
                                ys_s2, xs_s2 = np.nonzero(src_fg2)
                                loss = float(np.mean(dt[ys_s2, xs_s2] ** 2))
                        except Exception:
                            loss = float('inf')
                            shifted = warped
                else:
                    src_fg = (warped > 0)
                    if src_fg.sum() == 0:
                        loss = float('inf')
                        shifted = warped
                    else:
                        ys_s, xs_s = np.nonzero(src_fg)
                        loss = float(np.mean(dt[ys_s, xs_s] ** 2))
                        shifted = warped
                if loss < best_loss:
                    best_loss = loss
                    best_angle = a
                    best_warp = shifted
                if visualize:
                    disp = _build_overlay(tgt_img, warped)
                    disp_show = _scale_for_display(disp, max_dim=1280)
                    cv2.imshow(window_name, disp_show)
                    if save_steps:
                        cv2.imwrite(f"{out_prefix}_{step_i:03d}_ref_{a:03d}.png", disp)
                    key = cv2.waitKey(50) & 0xFF
                    if key in (27, ord('q')):
                        visualize = False
                    step_i += 1

        # store best overall
        params = dict(scale=scale, angle_deg=float(best_angle), tx=tx, ty=ty, flip_x=flip_x, flip_y=flip_y)
        # build combined M: rotation about target center then M0
        Rmat = cv2.getRotationMatrix2D((tgt_cx, tgt_cy), best_angle, 1.0)
        # combine: first apply M0 then rotation R; so total M = R * [M0; 0 0 1]
        M0_h = np.vstack([M0, [0.0, 0.0, 1.0]])
        R_h = np.vstack([Rmat, [0.0, 0.0, 1.0]])
        M_total = (R_h @ M0_h)[:2, :]

        if best is None or best_loss > best_loss and False:
            pass
        # choose by loss
        if best is None or best_loss < best[0]:
            best = (best_loss, M_total, params, best_warp)

    # if visualize, show final best warp
    if visualize and best is not None:
        final_warp = best[3]
        disp = _build_overlay(tgt_img, final_warp)
        disp_show = _scale_for_display(disp, max_dim=1280)
        cv2.imshow(window_name, disp_show)
        if save_steps:
            cv2.imwrite(f"{out_prefix}_final_best.png", disp)
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)

    if best is None:
        raise RuntimeError('Alignment failed for all flips')

    return best[1], best[2], best[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--point', '-p', type=str, default='colmap_projects/floor_1/png/density_bw.png')
    parser.add_argument('--floor', '-f', type=str, default='plan/20260428_112650_crop.jpg')
    parser.add_argument('--out', '-o', type=str, default='alignment_overlay.png')
    parser.add_argument('--max-sample', dest='max_sample', type=int, default=3000,
                        help='Maximum number of sample points to use from each mask (<=0 to use all)')
    parser.add_argument('--show-steps', dest='show_steps', action='store_true', help='Show intermediate step windows')
    parser.add_argument('--save-steps', dest='save_steps', action='store_true', help='Save intermediate step images')
    parser.add_argument('--step-prefix', dest='step_prefix', type=str, default='step', help='Prefix for saved step images')
    parser.add_argument('--no-center-align', dest='no_center_align', action='store_true', help='Do not center-align source to target before rotation (use translation search)')
    parser.add_argument('--floor-thresh', dest='floor_thresh', type=float, default=0.2, help='Normalized threshold for floor plan mask (use smaller to keep only almost black pixels)')
    args = parser.parse_args()

    point_path = Path(args.point)
    floor_path = Path(args.floor)
    if not point_path.exists() or not floor_path.exists():
        print('Default test files not found. Please provide --point and --floor paths.')
        print('Tried:', point_path, floor_path)
        sys.exit(1)

    print('Loading and extracting points...')
    src_pts, src_img, src_bin = extract_points_from_image(point_path)
    tgt_pts, tgt_img, tgt_bin = extract_points_from_image(floor_path, thresh=args.floor_thresh)

    print(f'Source points: {len(src_pts)}, Target (floor) points: {len(tgt_pts)}')

    best_loss = float('inf')
    best_res = None
    best_flip = (False, False)

    # Use mask-based center alignment with optional visualization
    try:
        M_total, params, loss = align_masks_by_center(src_bin, tgt_bin, tgt_img=tgt_img,
                                                     try_flips=True, coarse_step=5, refine=True,
                                                     visualize=args.show_steps, save_steps=args.save_steps,
                                                     out_prefix=args.step_prefix, center_align=not args.no_center_align)
        print('Mask-based alignment params:', params, 'loss:', loss)
        M = M_total
        # Build printed params for consistency
        s = params['scale']
        theta = np.radians(params['angle_deg'])
        tx = params['tx']
        ty = params['ty']
        flip_x = params['flip_x']
        flip_y = params['flip_y']
    except Exception as e:
        print('Mask-based alignment failed, falling back to point-based ICP:', e)
        for flip_x in (False, True):
            for flip_y in (False, True):
                try:
                    print(f'Trying flip_x={flip_x}, flip_y={flip_y} ...')
                    res = fit_similarity(src_pts, tgt_pts, flip_x, flip_y, max_sample=args.max_sample)
                    if res.fun < best_loss:
                        best_loss = float(res.fun)
                        best_res = res
                        best_flip = (flip_x, flip_y)
                except Exception as e:
                    print('Fit error for flip', flip_x, flip_y, e)

        if best_res is None:
            print('Failed to compute alignment')
            sys.exit(1)

        log_s, theta, tx, ty = best_res.x
        s = float(np.exp(log_s))
        flip_x, flip_y = best_flip
        M = build_affine_matrix(s, theta, tx, ty, flip_x, flip_y)

    # Print final transform parameters (s, theta in radians, tx, ty, flips)
    print('\nFinal transform:')
    print(f'  scale: {s:.6f}, rotation (deg): {np.degrees(theta):.3f}, tx: {tx:.2f}, ty: {ty:.2f}')
    print(f'  flip_x: {flip_x}, flip_y: {flip_y}')

    overlay = overlay_and_save(src_img, tgt_img, M, args.out)
    print('Saved overlay to', args.out)

    # show the result
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title('Overlay (red = point map projected on floor plan)')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()

