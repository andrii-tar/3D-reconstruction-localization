import torch
import numpy as np
from PIL import Image
from pathlib import Path
import torchvision.transforms.functional as TF

# Import the Regressor from the ACE repo
from ace_network import Regressor

import cv2
import matplotlib.pyplot as plt
import open3d as o3d


def load_transform(transform_path):
    """Load 4x4 transformation matrix from file."""
    if not Path(transform_path).exists():
        print(f"Warning: Transform file not found: {transform_path}")
        return None
    return np.loadtxt(transform_path)


def filter_central_region(points, percentile=95):
    """Filter points to keep only central region based on distance from centroid.
    
    Args:
        points: Nx3 array of 3D points
        percentile: Keep points within this percentile of distance (default 95)
    
    Returns:
        filtered_points: Points within the central region
        mask: Boolean mask of kept points
    """
    if points.shape[0] == 0:
        return points, np.ones(points.shape[0], dtype=bool)
    
    centroid = points.mean(axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)
    threshold = np.percentile(distances, percentile)
    mask = distances <= threshold
    
    filtered = points[mask]
    print(f"Filtered: {points.shape[0]} -> {filtered.shape[0]} points (keeping {percentile}% central region)")
    
    return filtered, mask


def project_to_xy(points):
    """Project 3D points to XY plane (drop Z coordinate).
    
    Args:
        points: Nx3 array of 3D points
    
    Returns:
        Nx2 array of XY coordinates
    """
    return points[:, :2]


def visualize_2d_projection(scene_points, camera_position, title="2D XY Projection"):
    """Create 2D matplotlib visualization of XY projection.
    
    Args:
        scene_points: Nx3 array of 3D points
        camera_position: 3D camera position
        title: Plot title
    """
    # Filter to central region
    points_filtered, _ = filter_central_region(scene_points, percentile=95)
    
    # Project to XY
    points_2d = project_to_xy(points_filtered)
    camera_2d = camera_position[:2]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(points_2d[:, 0], points_2d[:, 1], s=0.5, c='blue', alpha=0.3, label='Scene Points')
    ax.scatter(camera_2d[0], camera_2d[1], s=100, c='red', marker='*', label='Camera Location')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.axis('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def apply_transform_to_points(points, T):
    """Apply 4x4 transform to Nx3 points."""
    n = points.shape[0]
    ones = np.ones((n, 1))
    pts_h = np.hstack([points, ones])
    transformed = (T @ pts_h.T).T
    return transformed[:, :3]


def load_2d_transform(filepath):
    """Load 2D transform params from file."""
    if not Path(filepath).exists():
        print(f"Warning: 2D transform file not found: {filepath}")
        return None
    
    params = {}
    origin_x_img = None
    origin_y_img = None
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                try:
                    key, val = line.split('=', 1)
                    params[key.strip()] = float(val.strip())
                except:
                    pass
    
    # Also read origin from comment
    with open(filepath, 'r') as f:
        for line in f:
            if 'X=' in line and 'Y=' in line and 'origin' not in line.lower():
                import re
                m = re.search(r'X=([-\d.]+),\s*Y=([-\d.]+)', line)
                if m:
                    origin_x_img = float(m.group(1))
                    origin_y_img = float(m.group(2))
    
    return {
        'scale_x': params.get('scale_x', 1.0),
        'scale_y': params.get('scale_y', 1.0),
        'rotation_deg': params.get('rotation_deg', 0.0),
        'tx': params.get('tx', 0.0),
        'ty': params.get('ty', 0.0),
        'original_image_width': params.get('original_image_width', 3277),
        'original_image_height': params.get('original_image_height', 919),
        'output_width': params.get('output_width', 700),
        'output_height': params.get('output_height', 200),
        'origin_x_img': origin_x_img,
        'origin_y_img': origin_y_img
    }


def apply_2d_transform(x, y, transform):
    """Apply 2D transform to get floor plan coordinates."""
    # Scale
    x = x * transform['scale_x']
    y = y * transform['scale_y']
    
    # Rotate
    angle_rad = np.radians(transform['rotation_deg'])
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    x_new = x * cos_a - y * sin_a
    y_new = x * sin_a + y * cos_a
    
    # Translate
    x_new += transform['tx']
    y_new += transform['ty']
    
    return x_new, y_new


def coords_to_floorplan(world_x, world_y, transform):
    """Convert world (X, Y) to floor plan coordinates (0-700, 0-200).
    
    Args:
        world_x, world_y: 3D world coordinates
        transform: loaded 2D transform dict from load_2d_transform()
    
    Returns:
        (x, y) in floor plan (bottom-left origin)
    """
    # Get dimensions
    orig_w = transform.get('original_image_width', 3277)
    orig_h = transform.get('original_image_height', 919)
    out_w = transform.get('output_width', 700)
    out_h = transform.get('output_height', 200)
    
    # Scale factors
    scale_factor_w = out_w / orig_w
    scale_factor_h = out_h / orig_h
    
    # Scale transform params
    scale_x = transform['scale_x'] * scale_factor_w
    scale_y = transform['scale_y'] * scale_factor_h
    tx = transform['tx'] * scale_factor_w
    ty = transform['ty'] * scale_factor_h
    rotation = transform['rotation_deg']
    
    # Apply scaled transform
    x = world_x * scale_x
    y = world_y * scale_y
    
    angle_rad = np.radians(rotation)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    x_rot = x * cos_a - y * sin_a
    y_rot = x * sin_a + y * cos_a
    
    x_img = x_rot + tx
    y_img = y_rot + ty
    
    # Convert to floor plan coords (bottom-left origin)
    x_floor = x_img
    y_floor = out_h - y_img
    
    return x_floor, y_floor


def apply_transform_to_pose(rvec, tvec, T):
    """Apply transform to camera pose (rvec, tvec).
    
    T transforms points from old coords to new coords.
    For pose: new_pose = T @ old_pose
    """
    R_w2c, _ = cv2.Rodrigues(rvec)
    t_w2c = tvec.reshape(3, 1)
    
    # Current camera center in world: C = -R^T @ t
    C = -R_w2c.T @ t_w2c
    
    # Transform camera center to new coordinate system
    C_new = T[:3, :3] @ C + T[:3, 3].reshape(3, 1)
    
    # Compute new rotation (same as before, just in new coords)
    R_w2c_new = T[:3, :3] @ R_w2c
    
    # Convert back to Rodrigues
    rvec_new, _ = cv2.Rodrigues(R_w2c_new)
    
    # Compute new translation: t = -R @ C
    tvec_new = (-R_w2c_new @ C_new).ravel()
    
    return rvec_new, tvec_new


def visualize_3d_location(colmap_model_path, rvec, tvec, camera_matrix, width, height, transform_path=None, transform_2d_path=None):
    """
    Renders the room point cloud and places a red sphere + frustum at the camera position.
    """
    print("--- Step 6: Rendering 3D Scene ---")

    # Load 3D transform if provided
    T = None
    if transform_path:
        T = load_transform(transform_path)
        if T is not None:
            print(f"Loaded 3D transform from {transform_path}")
            print(f"Transform:\n{T}")
    
    # Load 2D transform if provided
    transform_2d = None
    if transform_2d_path:
        transform_2d = load_2d_transform(transform_2d_path)
        if transform_2d:
            print(f"Loaded 2D transform from {transform_2d_path}")
            print(f"2D transform: {transform_2d}")

    scene = o3d.geometry.PointCloud()
    pcd_found = False

    folder = Path(colmap_model_path)
    for ply_file in folder.glob("*.ply"):
        print(f"Loading point cloud: {ply_file.name}")
        scene = o3d.io.read_point_cloud(str(ply_file))
        
        # Apply transform to point cloud
        if T is not None:
            points = np.asarray(scene.points)
            points_transformed = apply_transform_to_points(points, T)
            scene.points = o3d.utility.Vector3dVector(points_transformed)
            print(f"Applied transform to {len(points)} points")
        
        pcd_found = True
        break

    if not pcd_found:
        print(f"❌ ERROR: No .ply file found in {colmap_model_path}")
        scene = o3d.geometry.PointCloud()
        points = np.random.uniform(-5, 5, (2000, 3))
        points[:, 2] = 0
        scene.points = o3d.utility.Vector3dVector(points)

    # Apply transform to camera pose if transform provided
    if T is not None:
        rvec, tvec = apply_transform_to_pose(rvec, tvec, T)
        print(f"Transformed camera pose: rvec={rvec.ravel()}, tvec={tvec}")

    # 1. Base OpenCV Pose (World-to-Camera)
    R_w2c, _ = cv2.Rodrigues(rvec)
    t_w2c = tvec.reshape(3, 1)

    # Calculate exact camera center in World space: C = -R^T * t
    camera_center = -np.dot(R_w2c.T, t_w2c)

    # Apply ACE Training Mean offset if applicable
    ace_training_mean = np.array([0.04, -0.03, 0.05]).reshape(3, 1)
    camera_center += ace_training_mean

    # CRITICAL FIX: If we shift the camera center, we MUST recalculate tvec
    # so the Open3D frustum moves with the red sphere!
    t_w2c_new = -np.dot(R_w2c, camera_center)

    # 2D XY Projection (after camera_center is defined)
    print("--- Creating 2D XY Projection ---")
    scene_points = np.asarray(scene.points)
    visualize_2d_projection(scene_points, camera_center.ravel(), "2D XY Projection (Top View)")

    # Open3D's visualize_camera expects the pure OpenCV Extrinsic Matrix (World-to-Camera)
    # NO OpenGL flip required!
    extrinsic_opencv = np.eye(4)
    extrinsic_opencv[:3, :3] = R_w2c
    extrinsic_opencv[:3, 3] = t_w2c_new.ravel()

    # Camera-to-World matrix just for drawing the RGB coordinate frame axes
    c2w_opencv = np.linalg.inv(extrinsic_opencv)

    # 2. Create the Marker (Red Sphere)
    marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    marker.paint_uniform_color([1, 0, 0])  # Red
    marker.translate(camera_center.ravel())

    # 3. Create the Coordinate Frame (RGB axes)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    frame.transform(c2w_opencv)

    # 4. Create the Frustum (Viewing Pyramid)
    frustum = o3d.geometry.LineSet.create_camera_visualization(
        view_width_px=int(width),
        view_height_px=int(height),
        intrinsic=camera_matrix,
        extrinsic=extrinsic_opencv
    )
    frustum.paint_uniform_color([0, 1, 0])  # Green pyramid

    # 5. Render
    print(f"Camera successfully placed at: \n{camera_center.ravel()}")
    o3d.visualization.draw_geometries([scene, marker, frame, frustum],
                                      window_name="Room 125 - Localization Result",
                                      front=[0.5, -0.5, -0.5],
                                      lookat=camera_center.ravel(),
                                      up=[0, 0, 1], # You may need to change this to [0, -1, 0] if the scene looks upside down
                                      zoom=0.8)

# Apply 2D transform to get floor plan coordinates
    if transform_2d:
        x_floor, y_floor = coords_to_floorplan(
            camera_center[0], 
            camera_center[1], 
            transform_2d
        )
        
        floorplan_w = transform_2d.get('output_width', 700)
        floorplan_h = transform_2d.get('output_height', 200)
        
        print(f"\n=== FLOOR PLAN COORDINATES ===")
        print(f"Camera position (bottom-left origin):")
        print(f"  X: {x_floor} (0=left, {floorplan_w}=right)")
        print(f"  Y: {y_floor} (0=bottom, {floorplan_h}=top)")
        
        # Save to file
        output_file = Path(colmap_model_path).parent / "camera_floorplan_coords.txt"
        with open(output_file, 'w') as f:
            f.write("# Camera position in floor plan coordinates\n")
            f.write(f"# Floor plan: bottom-left=(0,0), top-right=({floorplan_w},{floorplan_h})\n")
            f.write(f"x={x_floor}\n")
            f.write(f"y={y_floor}\n")
        print(f"Saved to {output_file}")
        
        if transform_2d.get('origin_x_floor') is not None:
            print(f"\n3D Model origin (0,0) at:")
            print(f"  X={transform_2d['origin_x_floor']}, Y={transform_2d['origin_y_floor']}")
        
        print(f"================================\n")


def visualize_results(image_path, points2D, points3D, rvec, tvec, camera_matrix, dist_coeffs, inliers):
    """
    Overlays predicted coordinates and reprojected points on the image.
    """
    # Load original image
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Project the 3D points predicted by ACE back to the image plane
    reprojected_pts, _ = cv2.projectPoints(points3D, rvec, tvec, camera_matrix, dist_coeffs)
    reprojected_pts = reprojected_pts.reshape(-1, 2)

    num_inliers = len(inliers) if inliers is not None else 0

    # Create figure
    plt.figure(figsize=(15, 10))

    # Subplot 1: Original Image with prediction heat map
    plt.subplot(1, 2, 1)
    plt.title("ACE Prediction Heatmap (3D Coordinates)")
    norm_pts = (points3D - points3D.min()) / (points3D.max() - points3D.min())
    plt.imshow(img)
    plt.scatter(points2D[:, 0], points2D[:, 1], c=norm_pts, s=1, alpha=0.5)

    # Subplot 2: Reprojection check
    plt.subplot(1, 2, 2)
    plt.title(f"Reprojection (Inliers: {num_inliers})")
    plt.imshow(img)

    # Draw a subset of points to avoid clutter
    indices = np.random.choice(len(reprojected_pts), min(500, len(reprojected_pts)), replace=False)
    plt.scatter(reprojected_pts[indices, 0], reprojected_pts[indices, 1], c='r', s=2, label='Reprojected')
    plt.legend()

    plt.tight_layout()
    plt.show()


def localize_with_ace(query_image_path, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Initialize Regressor
    print(f"--- Initializing Regressor ---")
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    regressor = Regressor(
        num_head_blocks=1,
        use_homogeneous=True,
        mean=mean
    )

    # 2. Load the Encoder weights separately
    encoder_path = Path(__file__).parent / "ace_encoder_pretrained.pt"
    print(f"--- Loading Encoder: {encoder_path.name} ---")
    regressor.encoder.load_state_dict(torch.load(encoder_path, map_location=device))

    # 3. Load and Fix the Head weights
    print(f"--- Loading Scene Map: {model_path.name} ---")
    state_dict = torch.load(model_path, map_location=device)

    fixed_head_state_dict = {}
    for key, value in state_dict.items():
        fixed_head_state_dict[f"heads.{key}"] = value

    regressor.load_state_dict(fixed_head_state_dict, strict=False)
    regressor = regressor.to(device)
    regressor.eval()

    # 2. Pre-process Image
    img = Image.open(query_image_path).convert('L')
    img = img.rotate(-90, expand=True)

    orig_w, orig_h = img.size

    target_h = 480
    target_w = int(orig_w * (target_h / orig_h))
    img_resized = img.resize((target_w, target_h), Image.LANCZOS)

    input_tensor = TF.to_tensor(img_resized).unsqueeze(0).to(device)

    # Update Normalization for Grayscale
    mean = torch.tensor([0.449]).view(1, 1, 1, 1).to(device)
    std = torch.tensor([0.226]).view(1, 1, 1, 1).to(device)
    input_tensor = (input_tensor - mean) / std

    # 3. Predict Scene Coordinates (SCR)
    print("--- Step 3: Predicting Scene Coordinates ---")
    with torch.no_grad():
        scene_coords = regressor(input_tensor)

    # 4. Prepare Data for PnP
    _, _, h_feat, w_feat = scene_coords.shape
    y, x = torch.meshgrid(torch.arange(h_feat), torch.arange(w_feat), indexing='ij')

    points2D = torch.stack(((x + 0.5) * 8 * (orig_w / target_w),
                            (y + 0.5) * 8 * (orig_h / target_h)), dim=-1).cpu().numpy().reshape(-1, 2)

    points3D = scene_coords.squeeze().permute(1, 2, 0).cpu().numpy().reshape(-1, 3)

    # 5. Solve Pose with OpenCV RANSAC
    print("--- Step 5: Solving Pose with OpenCV RANSAC ---")

    focal_px = 1.2 * max(orig_w, orig_h)
    # Try a more standard focal length ratio
    focal_px = 0.8 * max(orig_w, orig_h)
    camera_matrix = np.array([
        [focal_px, 0, orig_w / 2],
        [0, focal_px, orig_h / 2],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # Ensure arrays are contiguous float32 for OpenCV
    obj_pts = np.ascontiguousarray(points3D, dtype=np.float32)
    img_pts = np.ascontiguousarray(points2D, dtype=np.float32)

    # Run SolvePnPRansac with more forgiving parameters
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj_pts,
        img_pts,
        camera_matrix,
        dist_coeffs,
        reprojectionError=30.0,  # Increased from 10.0 to 30.0 to allow for focal length errors
        iterationsCount=5000,  # Give RANSAC more chances to find a good subset
        confidence=0.9,
        flags=cv2.SOLVEPNP_SQPNP  # SQPNP is often more robust to noise than EPNP
    )
    print("success:", success, "inliers:", len(inliers))
    if success and inliers is not None and len(inliers) > 15:
        print(f"\n✅ SUCCESS!")
        print(f"Number of Inliers: {len(inliers)} out of {len(img_pts)}")
        print(f"Rotation Vector (rvec):\n{rvec}")
        print(f"Translation Vector (tvec):\n{tvec}")

        # Visualize 2D results
        visualize_results(query_image_path, points2D, points3D, rvec, tvec, camera_matrix, dist_coeffs, inliers)

        # 3D Visualization
        colmap_path = r"E:\colmap_project_export\colmap\first_floor_and_stairs"
        transform_path = r"E:\colmap_project_export\artifacts\floor_1\transform.txt"
        transform_2d_path = r"E:\colmap_project_export\artifacts\floor_1\transform_2d.txt"
        visualize_3d_location(colmap_path, rvec, tvec, camera_matrix, orig_w, orig_h, transform_path, transform_2d_path)
    else:
        print("\n❌ Localization failed. RANSAC could not find a valid pose from the ACE predictions.")


if __name__ == "__main__":
    # trained_model = Path(r"D:\Projects\computer_vision\3d-reconstruction\3d_reconstruction_py\ace_model\room_125_v2.pt")
    trained_model = Path(r"F:\ace_dataset\ff_stairs_model\model_ff_stairs_v1_e_12.pt")
    query_img = Path(
        r"D:\Projects\computer_vision\3d-reconstruction\3d_reconstruction_py\fast-loc\Room_125\20260428_111252.jpg")
    localize_with_ace(query_img, trained_model)


# python -W ignore train_ace.py "D:\Projects\computer_vision\3d-reconstruction\ace_data\room_125" "D:\Projects\computer_vision\3d-reconstruction\3d_reconstruction_py\ace_model\room_125_v2.pt" --image_resolution 480 --epochs 12 --num_head_blocks 1