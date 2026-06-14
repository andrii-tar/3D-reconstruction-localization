#!/usr/bin/env python3
r"""
Convert 3D coordinates to floor plan pixel coordinates.

Usage:
    python coords_to_floorplan.py x y [z]
    
Or use as module:
    from coords_to_floorplan import transform_to_floorplan
    x, y = transform_to_floorplan(world_x, world_y)
"""

import argparse
import numpy as np
from pathlib import Path


def load_transform(filepath):
    """Load 2D transform from file."""
    if not Path(filepath).exists():
        print(f"Error: Transform file not found: {filepath}")
        return None
    
    params = {}
    origin_x_img = origin_y_img = origin_x_floor = origin_y_floor = None
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                try:
                    key, val = line.split('=', 1)
                    params[key.strip()] = float(val.strip())
                except:
                    pass
    
    # Parse origin from comments
    with open(filepath, 'r') as f:
        for line in f:
            if 'Image space' in line:
                import re
                m = re.search(r'X=([-\d.]+),\s*Y=([-\d.]+)', line)
                if m:
                    origin_x_img = float(m.group(1))
                    origin_y_img = float(m.group(2))
            if 'Floor plan' in line and 'origin' in line:
                import re
                m = re.search(r'X=([-\d.]+),\s*Y=([-\d.]+)', line)
                if m:
                    origin_x_floor = float(m.group(1))
                    origin_y_floor = float(m.group(2))
    
    return {
        'scale_x': params.get('scale_x', 1.0),
        'scale_y': params.get('scale_y', 1.0),
        'rotation_deg': params.get('rotation_deg', 0.0),
        'tx': params.get('tx', 0.0),
        'ty': params.get('ty', 0.0),
        'floorplan_width': params.get('floorplan_width', 700),
        'floorplan_height': params.get('floorplan_height', 200),
        'origin_x_img': origin_x_img,
        'origin_y_img': origin_y_img,
        'origin_x_floor': origin_x_floor,
        'origin_y_floor': origin_y_floor
    }


def transform_to_floorplan(world_x, world_y, transform):
    """Convert world (X, Y) to floor plan pixel coordinates (bottom-left origin).
    
    Args:
        world_x, world_y: 3D world coordinates
        transform: loaded transform dictionary
    
    Returns:
        (x, y) in floor plan (bottom-left origin, pixels)
    """
    # Apply 2D transform (scale, rotate, translate)
    scale_x = transform['scale_x']
    scale_y = transform['scale_y']
    rotation = transform['rotation_deg']
    tx = transform['tx']
    ty = transform['ty']
    
    # Scale
    x = world_x * scale_x
    y = world_y * scale_y
    
    # Rotate
    angle_rad = np.radians(rotation)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    x_rot = x * cos_a - y * sin_a
    y_rot = x * sin_a + y * cos_a
    
    # Translate to image space
    x_img = x_rot + tx
    y_img = y_rot + ty
    
    # Convert to floor plan coords (bottom-left origin)
    floor_h = transform['floorplan_height']
    x_floor = x_img
    y_floor = floor_h - y_img
    
    return x_floor, y_floor


def main():
    parser = argparse.ArgumentParser(description='Convert world coords to floor plan')
    parser.add_argument('x', type=float, help='World X')
    parser.add_argument('y', type=float, help='World Y')
    parser.add_argument('z', type=float, nargs='?', default=0, help='World Z (ignored, for XY plane)')
    parser.add_argument('--transform', '-t', default='transform_2d.txt', 
                        help='Path to transform_2d.txt')
    args = parser.parse_args()
    
    transform = load_transform(args.transform)
    if transform is None:
        return
    
    x_floor, y_floor = transform_to_floorplan(args.x, args.y, transform)
    
    print(f"World: ({args.x}, {args.y})")
    print(f"Floor plan (bottom-left origin):")
    print(f"  X: {x_floor:.2f} (0=left, {transform['floorplan_width']}=right)")
    print(f"  Y: {y_floor:.2f} (0=bottom, {transform['floorplan_height']}=top)")
    
    if transform.get('origin_x_floor') is not None:
        print(f"\nModel origin (0,0) at: X={transform['origin_x_floor']:.2f}, Y={transform['origin_y_floor']:.2f}")


if __name__ == '__main__':
    main()