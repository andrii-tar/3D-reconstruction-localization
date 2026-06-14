#!/usr/bin/env python3
"""
Simple runner to extract points from the two default images and save visualizations.
Saves files into the current working directory:
 - extracted_src_mask.png
 - extracted_src_scatter.png
 - extracted_tgt_mask.png
 - extracted_tgt_scatter.png

Run from the project directory.
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

import align_pointmap_to_floorplan as ap


def save_mask_overlay(img, bin_img, out_path):
    if len(img.shape) == 2:
        color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        color = img.copy()
    mask = bin_img.astype(bool)
    color[mask] = [0, 0, 255]
    cv2.imwrite(str(out_path), color)


def save_scatter(pts, img_shape, out_path):
    h, w = img_shape[0], img_shape[1]
    # Use aspect='equal' so scatter respects image aspect ratio
    fig = plt.figure(figsize=(w/100, h/100), dpi=100)
    ax = fig.add_subplot(111)
    ax.scatter(pts[:, 0], pts[:, 1], s=0.5, c='k')
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)


def main():
    src_path = Path('colmap_projects/floor_1/png/density_bw.png')
    tgt_path = Path('plan/20260428_112650_crop.jpg')
    if not src_path.exists() or not tgt_path.exists():
        print('Default files not found; please run from project root where files exist.')
        return

    src_pts, src_img, src_bin = ap.extract_points_from_image(src_path)
    tgt_pts, tgt_img, tgt_bin = ap.extract_points_from_image(tgt_path)

    print('Source points:', len(src_pts))
    print('Target points:', len(tgt_pts))
    print('Sample source points (first 10):')
    print(src_pts[:10])

    save_mask_overlay(src_img, src_bin, 'extracted_src_mask.png')
    save_scatter(src_pts, src_img.shape, 'extracted_src_scatter.png')
    save_mask_overlay(tgt_img, tgt_bin, 'extracted_tgt_mask.png')
    save_scatter(tgt_pts, tgt_img.shape, 'extracted_tgt_scatter.png')

    print('Saved: extracted_src_mask.png, extracted_src_scatter.png, extracted_tgt_mask.png, extracted_tgt_scatter.png')


if __name__ == '__main__':
    main()

