import pycolmap
from pathlib import Path
import numpy as np
import shutil
from tqdm import tqdm


def convert(colmap_model_path, ace_dataset_path, image_dir):
    colmap_model_path = Path(colmap_model_path)
    ace_dataset_path = Path(ace_dataset_path)
    image_dir = Path(image_dir)

    # Create ACE structure
    rgb_out = ace_dataset_path / "train" / "rgb"
    pose_out = ace_dataset_path / "train" / "poses"
    calib_out = ace_dataset_path / "train" / "calibration"

    rgb_out.mkdir(parents=True, exist_ok=True)
    pose_out.mkdir(parents=True, exist_ok=True)
    calib_out.mkdir(parents=True, exist_ok=True)

    reconstruction = pycolmap.Reconstruction(colmap_model_path)

    print(f"Converting {len(reconstruction.images)} images...")

    # iterate with progress bar
    images = list(reconstruction.images.items())
    # Get the focal length from the first camera (assuming fixed intrinsics)
    cam = reconstruction.cameras[next(iter(reconstruction.cameras))]
    # ACE usually wants the mean focal length for SCR
    focal_length = cam.params[0]

    for image_id, image in tqdm(images, total=len(images), desc="Converting images"):
        # 1. Copy Image
        # image.name may include subdirectory components (e.g. "IMG_1293_1080p\\frame_0560.png").
        # Preserve that structure under ace_dataset_path/train/rgb and ensure parent dirs exist.
        src_image = image_dir / Path(image.name)
        dst_image = rgb_out / Path(image.name)
        # print(image_dir, src_image, rgb_out)
        # print(src_image.exists())
        if src_image.exists():
            # Make sure destination subdirectories exist before copying
            dst_image.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src_image, dst_image)
        else:
            print(f"Warning: Image {image.name} not found in {image_dir}")
            continue

        # 2. Convert Pose (W2C to C2W)
        w2c = np.eye(4)
        pose = image.cam_from_world()
        w2c[:3, :3] = pose.rotation.matrix()
        w2c[:3, 3] = pose.translation

        # Invert to get Camera-to-World (C2W) for ACE
        c2w = np.linalg.inv(w2c)

        # 3. Save Pose File
        # Preserve subdirectory structure for pose/calibration files too so they match the image layout.
        pose_fn_path = Path(image.name).with_suffix('.txt')
        pose_out_path = pose_out / pose_fn_path
        pose_out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(pose_out_path, c2w)

        # 4. Save Calibration File (One per image for this ACE version)
        # ACE expects the focal length in a text file matching the image name
        calib_out_path = calib_out / pose_fn_path
        calib_out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(calib_out_path, [focal_length])

    print(f"Done! Data ready at: {ace_dataset_path}")


if __name__ == "__main__":
    # Ensure these paths match your local setup
    convert(
        colmap_model_path=r"F:\colmap_second_floor\second_floor_full",
        ace_dataset_path=r"F:\ace_dataset\sf_stairs",
        image_dir=r"F:\colmap_second_floor\IMG_1297_full"
    )