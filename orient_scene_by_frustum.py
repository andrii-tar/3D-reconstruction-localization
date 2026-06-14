#!/usr/bin/env python3
r"""
orient_scene_by_frustum.py

Load a COLMAP sparse reconstruction and orient it such that most camera frames
are perpendicular to the floor (XY coordinate plane). Uses the camera viewing
direction (frustum/look direction) to determine orientation.

The key insight: In indoor scenes, cameras typically look horizontally (perpendicular
to floor). The camera's "up" vector (or the cross product of right and forward)
should align with the floor normal.

Usage:
    python orient_scene_by_frustum.py path/to/reconstruction --output_dir path/to/output

Outputs:
    - 4x4 transformation matrix saved as transform.txt
"""

import argparse
import pathlib
import sys
import numpy as np
import math
from scipy.spatial.transform import Rotation as R_scipy

try:
    import pycolmap
except Exception:
    print("Error: pycolmap not installed. Please install pycolmap.")
    raise


def get_camera_orientations(reconstruction):
    """Extract camera positions and viewing directions from reconstruction."""
    cameras_data = []
    failed_count = 0

    for idx, (img_id, img) in enumerate(reconstruction.images.items()):
        # Check if image has pose
        has_pose = getattr(img, 'has_pose', False)
        if not has_pose:
            failed_count += 1
            continue

        center = None
        R = None

        # Debug counter for first few images
        debug_img = idx < 3

        # Method 1: Try cam_from_world (Rigid3d object)
        cfw = getattr(img, 'cam_from_world', None)
        if cfw is not None and debug_img:
            print(f"img {img_id}: cfw exists, callable={callable(cfw)}")
        if cfw is not None:
            try:
                # It's a Rigid3d object, get rotation and translation
                if callable(cfw):
                    pose = cfw()
                else:
                    pose = cfw
                
                # Get rotation and translation - use matrix() for proper 3x3
                M = np.array(pose.matrix())
                if debug_img:
                    print(f"  M shape: {M.shape}")
                
                if M.shape == (3, 4):
                    R = M[:3, :3]
                    t = M[:3, 3]
                    center = -R.T @ t
                elif M.shape == (4, 4):
                    R = M[:3, :3]
                    t = M[:3, 3]
                    center = -R.T @ t
                else:
                    # Fallback to rotation/translation
                    R = np.array(pose.rotation)
                    if R.ndim == 0:
                        # rotation is likely a quaternion or other representation
                        # Skip this pose
                        raise ValueError("Cannot extract 3x3 rotation")
                    t = np.array(pose.translation)
                    center = -R.T @ t
            except Exception as e:
                if debug_img:
                    print(f"  Exception: {e}")

        # Method 2: Try qvec + tvec (older API)
        if R is None:
            try:
                qvec = getattr(img, 'qvec', None)
                tvec = getattr(img, 'tvec', None)
                if qvec is not None and tvec is not None:
                    qvec = np.array(qvec)
                    tvec = np.array(tvec)
                    if np.abs(qvec).sum() > 1e-6 and np.abs(tvec).sum() > 1e-6:
                        if hasattr(pycolmap, 'qvec2rotmat'):
                            R = pycolmap.qvec2rotmat(qvec)
                            center = -R.T @ tvec
            except Exception:
                pass

        # Method 3: Try projection_center
        if center is None:
            try:
                pc = getattr(img, 'projection_center', None)
                if pc is not None:
                    if callable(pc):
                        center = np.array(pc())
                    else:
                        center = np.array(pc)
            except Exception:
                pass

        # Method 4: Try frame.rig_from_world() or frame.sensor_from_world()
        if R is None or center is None:
            try:
                frame = getattr(img, 'frame', None)
                if frame is not None:
                    # Try sensor_from_world with sensor_id from data_ids
                    if hasattr(frame, 'sensor_from_world') and callable(frame.sensor_from_world):
                        try:
                            # Get sensor_id from data_ids (first one with CAMERA type)
                            data_ids = list(frame.data_ids)
                            sensor_id = None
                            for dt in data_ids:
                                if len(dt) >= 2 and str(dt[0]) == 'CAMERA':
                                    sensor_id = dt[1]
                                    break
                            if sensor_id is not None:
                                pose = frame.sensor_from_world(sensor_id)
                                if hasattr(pose, 'rotation') and hasattr(pose, 'translation'):
                                    R = np.array(pose.rotation)
                                    if R.ndim == 0:
                                        R = R.reshape(3, 3)
                                    t = np.array(pose.translation)
                                    if t.ndim == 0:
                                        t = t.reshape(3)
                                    center = -R.T @ t
                        except Exception:
                            pass

                    # Try rig_from_world
                    if R is None:
                        try:
                            if hasattr(frame, 'rig_from_world') and callable(frame.rig_from_world):
                                pose = frame.rig_from_world()
                                if hasattr(pose, 'rotation') and hasattr(pose, 'translation'):
                                    R = np.array(pose.rotation)
                                    if R.ndim == 0:
                                        R = R.reshape(3, 3)
                                    t = np.array(pose.translation)
                                    if t.ndim == 0:
                                        t = t.reshape(3)
                                    center = -R.T @ t
                        except Exception:
                            pass
            except Exception:
                pass

        if R is None or center is None:
            failed_count += 1
            continue

        forward = -R[2, :]
        up = R[1, :]
        right = R[0, :]

        name = img.name if hasattr(img, 'name') else str(img_id)
        
        # Debug first few
        if len(cameras_data) < 3:
            print(f"  Added camera {img_id}: center={center[:3]}, forward={forward[:3]}, up={up[:3]}")
        
        cameras_data.append((img_id, name, np.asarray(center), forward, up, right))

    if failed_count > 0:
        print(f"Note: {failed_count} images have no valid pose (unregistered)")
    return cameras_data


def estimate_floor_normal(cameras_data):
    """Estimate floor normal from camera up vectors.

    For indoor scenes, most camera up vectors should point toward the ceiling
    (or floor normal points down). We use the median to be robust to outliers.
    """
    up_vectors = np.array([c[4] for c in cameras_data])
    
    # Remove any invalid vectors
    valid = ~np.isnan(up_vectors).any(axis=1) & (np.linalg.norm(up_vectors, axis=1) > 1e-6)
    up_vectors = up_vectors[valid]
    
    if len(up_vectors) == 0:
        return np.array([0, 0, 1]), np.array([0, 0, 1])
    
    mean_up = np.mean(up_vectors, axis=0)
    mean_up = mean_up / np.linalg.norm(mean_up)

    median_up = np.median(up_vectors, axis=0)
    norm = np.linalg.norm(median_up)
    if norm < 1e-6:
        median_up = mean_up
    else:
        median_up = median_up / norm

    return mean_up, median_up


def rotation_matrix_from_vectors(a, b):
    """Compute rotation matrix that rotates vector a to vector b."""
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


def find_best_rotation_to_xy_plane(cameras_data):
    """Find rotation that makes most camera forward directions lie in XY plane.

    After rotation, camera forward vectors should be mostly horizontal (Z ~= 0),
    meaning cameras are looking parallel to the floor.

    Strategy:
    1. Compute average forward direction
    2. Find rotation that makes this horizontal (z component = 0)
    3. Apply additional refinement if needed
    """
    forward_vectors = np.array([c[3] for c in cameras_data])

    avg_forward = np.mean(forward_vectors, axis=0)
    avg_forward = avg_forward / np.linalg.norm(avg_forward)

    horizontal_proj = np.array([avg_forward[0], avg_forward[1], 0])
    horizontal_proj = horizontal_proj / (np.linalg.norm(horizontal_proj) + 1e-8)

    R1 = rotation_matrix_from_vectors(avg_forward, horizontal_proj)

    rotated_forwards = (R1 @ forward_vectors.T).T
    avg_z_after = np.mean(np.abs(rotated_forwards[:, 2]))

    if avg_z_after > 0.1:
        z_components = forward_vectors[:, 2]
        median_z = np.median(z_components)
        estimated_up = np.array([0, 0, 1]) if median_z > 0 else np.array([0, 0, -1])

        mean_up, _ = estimate_floor_normal(cameras_data)
        if np.dot(mean_up, estimated_up) < 0:
            estimated_up = -estimated_up

        R2 = rotation_matrix_from_vectors(mean_up, estimated_up)
        return R2

    return R1


def main():
    p = argparse.ArgumentParser(
        description='Orient scene using camera frustum directions so frames are perpendicular to floor'
    )
    p.add_argument('reconstruction', type=pathlib.Path,
                   help='Path to COLMAP sparse reconstruction (cameras.bin, images.bin, points3D.bin)')
    p.add_argument('--output_dir', type=pathlib.Path, default=None,
                   help='Output directory (default: reconstruction parent)')
    p.add_argument('--method', choices=['up_vector', 'forward_horizontal'], default='up_vector',
                   help='Method for orientation: up_vector uses camera up direction, forward_horizontal ensures forward is in XY plane')
    p.add_argument('--set_floor', action='store_true',
                   help='Translate so minimum Z becomes 0 (floor at Z=0)')
    p.add_argument('--invert_up', action='store_true',
                   help='Invert the up vector (flip floor normal)')
    args = p.parse_args()

    rec_path = args.reconstruction
    if not rec_path.exists():
        print('Reconstruction not found:', rec_path)
        sys.exit(1)

    out_dir = args.output_dir if args.output_dir else rec_path.parent
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('Loading reconstruction from', rec_path)
    reconstruction = pycolmap.Reconstruction(str(rec_path))

    print(f'Total images: {len(reconstruction.images)}')
    print(f'Total 3D points: {len(reconstruction.points3D)}')
    print(f'Total cameras: {len(reconstruction.cameras)}')

    # Debug: check if images have valid qvec/tvec
    valid_poses = 0
    zero_qvec = zero_tvec = 0
    for img_id, img in reconstruction.images.items():
        qvec = getattr(img, 'qvec', None)
        tvec = getattr(img, 'tvec', None)
        if qvec is not None and tvec is not None:
            qarr = np.array(qvec)
            tarr = np.array(tvec)
            if np.abs(qarr).sum() > 1e-6 and np.abs(tarr).sum() > 1e-6:
                valid_poses += 1
            elif np.abs(qarr).sum() <= 1e-6:
                zero_qvec += 1
            elif np.abs(tarr).sum() <= 1e-6:
                zero_tvec += 1

    print(f'\nPose status:')
    print(f'  Valid poses: {valid_poses}')
    print(f'  Zero qvec: {zero_qvec}')
    print(f'  Zero tvec: {zero_tvec}')

    # Check has_pose on images
    images_with_pose = sum(1 for img in reconstruction.images.values() if getattr(img, 'has_pose', False))
    print(f'  has_pose=True: {images_with_pose}')

    if len(reconstruction.points3D) == 0 and len(reconstruction.images) > 0:
        print('\nWARNING: Reconstruction has images but no 3D points!')
        print('This likely means images were imported but never triangulated.')
        print('Run COLMAP mapper to create sparse reconstruction first:')
        print('  colmap mapper --image_path <path> --database_path <path> --output_path <path>')

    # Debug: check first image structure
    if reconstruction.images:
        sample_img = next(iter(reconstruction.images.values()))
        print(f'\nSample image attributes:')
        for attr in dir(sample_img):
            if not attr.startswith('_'):
                val = getattr(sample_img, attr, None)
                if not callable(val):
                    print(f'  {attr}: {type(val).__name__}')

    cameras_data = get_camera_orientations(reconstruction)
    if len(cameras_data) < 3:
        print('Need at least 3 cameras. Found:', len(cameras_data))
        sys.exit(2)

    print(f'Loaded {len(cameras_data)} cameras')

    # Debug: check some sample orientations
    sample_idx = min(5, len(cameras_data))
    for i in range(sample_idx):
        c = cameras_data[i]
        print(f"  Sample {i}: center={c[2][:3]}, forward={c[3]}, up={c[4]}")

    # Get all forward and up vectors
    forwards = np.array([c[3] for c in cameras_data])
    ups = np.array([c[4] for c in cameras_data])
    centers = np.array([c[2] for c in cameras_data])
    
    # Filter valid
    valid_f = ~np.isnan(forwards).any(axis=1)
    valid_u = ~np.isnan(ups).any(axis=1)
    forwards = forwards[valid_f]
    ups = ups[valid_u]
    
    print(f'Forward vectors sample (first 5):\n{forwards[:5]}')
    
    # Simple approach: find rotation that makes forward vectors as horizontal as possible
    # We want to minimize sum of (forward_z)^2 across all cameras
    
    # Method: SVD on forward vectors to find primary horizontal direction
    # Then construct rotation that aligns this to X-axis
    
    # Compute average forward
    avg_forward = np.mean(forwards, axis=0)
    print(f'Average forward: {avg_forward}')
    
    # Project to XY - this is where cameras are looking on average
    horizontal_avg = np.array([avg_forward[0], avg_forward[1], 0])
    horizontal_avg = horizontal_avg / (np.linalg.norm(horizontal_avg) + 1e-8)
    
    # Also compute average up
    avg_up = np.mean(ups, axis=0)
    avg_up = avg_up / (np.linalg.norm(avg_up) + 1e-8)
    print(f'Average up: {avg_up}')
    
    # Build rotation: 
    # - First align avg_forward to horizontal plane
    # - Then align avg_up to vertical
    
    # Rotation 1: align forward to horizontal version of itself
    R1 = rotation_matrix_from_vectors(avg_forward, horizontal_avg)
    
    # Apply and check
    forwards_r1 = (R1 @ forwards.T).T
    ups_r1 = (R1 @ ups.T).T
    print(f'After R1 - forward Z: {np.mean(forwards_r1[:, 2]):.4f}, up Z: {np.mean(ups_r1[:, 2]):.4f}')
    
    # Rotation 2: align average up to Z axis
    rotated_up = R1 @ avg_up
    R2 = rotation_matrix_from_vectors(rotated_up, np.array([0, 0, 1]))
    
    R = R2 @ R1
    
    # Final check
    forwards_rot = (R @ forwards.T).T
    ups_rot = (R @ ups.T).T
    print(f'After full rotation:')
    print(f'  Avg forward: {np.mean(forwards_rot, axis=0)}')
    print(f'  Avg up: {np.mean(ups_rot, axis=0)}')
    print(f'  Forward Z (should be ~0): {np.mean(forwards_rot[:, 2]):.4f}')
    print(f'  Up Z (should be ~1): {np.mean(ups_rot[:, 2]):.4f}')
    
    # Try to optimize further with scipy - always apply
    def loss_fn(angles):
        rot = R_scipy.from_euler('xyz', angles)
        R_opt = rot.as_matrix()
        f_rot = (R_opt @ forwards.T).T
        u_rot = (R_opt @ ups.T).T
        # Minimize forward Z component and make up vertical
        cost = np.sum(f_rot[:, 2]**2) + np.sum((u_rot[:, 2] - 1)**2)
        return cost
    
    from scipy.optimize import minimize
    # Start from current R's Euler angles
    current_rot = R_scipy.from_matrix(R)
    initial_angles = current_rot.as_euler('xyz')
    print(f'Initial Euler angles: {initial_angles}')
    
    result = minimize(loss_fn, initial_angles, method='L-BFGS-B')
    print(f'Optimization result: success={result.success}, fun={result.fun:.4f}')
    
    R_opt = R_scipy.from_euler('xyz', result.x).as_matrix()
    R = R_opt
    
    forwards_rot = (R @ forwards.T).T
    ups_rot = (R @ ups.T).T
    print(f'After optimization:')
    print(f'  Forward Z (should be ~0): {np.mean(forwards_rot[:, 2]):.4f}')
    print(f'  Up Z (should be ~1): {np.mean(ups_rot[:, 2]):.4f}')
    
    print('Rotation matrix:\n', R)

    centers = np.array([c[2] for c in cameras_data])
    forwards = np.array([c[3] for c in cameras_data])
    ups = np.array([c[4] for c in cameras_data])

    centers_rot = (R @ centers.T).T
    forwards_rot = (R @ forwards.T).T
    ups_rot = (R @ ups.T).T

    avg_forward_after = np.mean(forwards_rot, axis=0)
    print('Average forward after rotation:', avg_forward_after)
    print('  Z component (should be ~0 if horizontal):', avg_forward_after[2])

    trans = np.zeros(3)
    if args.set_floor:
        min_z = np.min(centers_rot[:, 2])
        trans[2] = -min_z
        print('Translating by Z =', trans[2], 'to set floor at Z=0')

    centers_final = centers_rot + trans

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = trans

    tf_path = out_dir / 'transform.txt'
    np.savetxt(tf_path, T, fmt='%.8f')
    print('Saved transformation matrix to', tf_path)

    pts3d = []
    for pid, p in reconstruction.points3D.items():
        pts3d.append(np.array(p.xyz, dtype=float))
    pts3d = np.vstack(pts3d) if pts3d else np.zeros((0, 3))

    if pts3d.shape[0] > 0:
        pts3d_rot = (R @ pts3d.T).T + trans
        ply_path = out_dir / 'oriented_points.ply'
        header = f"""ply
format ascii 1.0
element vertex {pts3d_rot.shape[0]}
property float x
property float y
property float z
end_header
"""
        with open(ply_path, 'w') as f:
            f.write(header)
            for p in pts3d_rot:
                f.write(f'{p[0]} {p[1]} {p[2]}\n')
        print('Saved oriented points to', ply_path)

    print('\nTransformation matrix (4x4):')
    print(T)
    print('\nTo apply this transform to COLMAP model, use:')
    print('  colmap model_transformer --input_path <path> --output_path <path> --transform_path', tf_path)


if __name__ == '__main__':
    main()