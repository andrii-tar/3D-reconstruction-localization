#!/usr/bin/env python3
"""Apply transformation to COLMAP sparse model."""

import argparse
import numpy as np
import pycolmap

def apply_transform(reconstruction, T):
    """Apply 4x4 transform to all points and cameras."""
    R = T[:3, :3]
    t = T[:3, 3]
    
    # Transform 3D points
    for pid in reconstruction.points3D:
        pt = np.array(reconstruction.points3D[pid].xyz)
        pt_new = R @ pt + t
        reconstruction.points3D[pid].xyz = tuple(pt_new)
    
    # Transform camera poses
    for img_id, img in reconstruction.images.items():
        if not getattr(img, 'has_pose', False):
            continue
        
        # Get current pose
        cfw = getattr(img, 'cam_from_world', None)
        if cfw is None:
            continue
            
        pose = cfw() if callable(cfw) else cfw
        M = np.array(pose.matrix())
        
        if M.shape == (3, 4):
            Rcw = M[:3, :3]
            tcw = M[:3, 3]
        else:
            continue
        
        # Transform: new_T = T @ old_T
        # Camera center in world: C = -Rcw.T @ tcw
        # After transform: C' = R @ C + t
        # New pose: C' = -Rcw'.T @ tcw'
        # => Rcw'.T @ tcw' = -(R @ (-Rcw.T @ tcw) + t)
        # => Rcw'.T @ tcw' = R @ Rcw.T @ tcw - R @ t
        
        R_new = R @ Rcw
        t_new = R @ (-Rcw.T @ tcw) + t
        
        # Create new Rigid3d pose
        from pycolmap._core import Rigid3d, RotationMatrix, EigenVector3d
        new_pose = Rigid3d(RotationMatrix(R_new), EigenVector3d(t_new))
        
        # Set the new pose
        if hasattr(img, 'set_cam_from_world'):
            img.set_cam_from_world(new_pose)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('input', help='Input COLMAP model')
    p.add_argument('output', help='Output COLMAP model')
    p.add_argument('transform', help='4x4 transform matrix .txt file')
    args = p.parse_args()
    
    print(f"Loading model from {args.input}")
    rec = pycolmap.Reconstruction(str(args.input))
    
    print(f"Loading transform from {args.transform}")
    T = np.loadtxt(args.transform)
    print(f"Transform:\n{T}")
    
    print("Applying transform...")
    apply_transform(rec, T)
    
    print(f"Writing to {args.output}")
    rec.write(str(args.output))
    
    print("Done!")

if __name__ == '__main__':
    main()