import math
import os

import cv2
import numpy as np
import torch

from vggt.models.vggt import VGGT
from vggt.utils.geometry import (
    closed_form_inverse_se3,
    depth_to_world_coords_points,
)
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

print("################################# VGGT Inference ##################################\n")




device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

model.eval()

# Load and preprocess example images (replace with your own image paths)
image_names = []
for img in os.listdir("images"):
    if img.endswith((".jpg", ".jpeg", ".png")):
        image_names.append(os.path.join("images", img))
image_names = sorted(image_names)
print(f"image names: {image_names}")
images = []

for img in image_names:
    im = cv2.imread(img)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    h, w, _ = im.shape
    print(f"Image height: {h}, width: {w}")

    im.resize((int(w / 2), int(h / 2)))
    images.append(img)
    print(f"Image {img} resized to: {int(h / 2)}x{int(w / 2)}")

images = load_and_preprocess_images(images).to(device)


with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        images = images[None]  # [B, S, C, H, W]
        aggregated_tokens_list, ps_idx = model.aggregator(images)

    pose_enc = model.camera_head(aggregated_tokens_list)[-1]

    # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        pose_enc, images.shape[-2:]
    )  # (B, S, 3, 4) and (B, S, 3, 3)

    B, V = extrinsic.shape[:2]  # [1, 6, 3, 4]
    extrinsic_homo = torch.eye(4, device=device).repeat(B, V, 1, 1)  # [1, 6, 4, 4]
    extrinsic_homo[:, :, :3, :] = extrinsic
    transformation = torch.eye(4, device=device)
    transformation = torch.tensor(
        [
            [1, 0, 0, 0],  # x right -> x right
            [0, 0, -1, 0],  # y down -> z up
            [0, -1, 0, 0],  # z forward -> y towards (forward translation)
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
        device=extrinsic.device,
    )

    transformation = transformation[None, None, :, :]  # [1, 1, 4, 4]
    extrinsic_homo = extrinsic_homo @ transformation  # [B, S, 4, 4]
    extrinsic = extrinsic_homo[:, :, :3, :]

    # Predict Depth Maps
    depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    # remove batch dimension
    depth_map = depth_map[0]  # [S, H, W, 1]
    extrinsic = extrinsic[0]  # [S, 3, 4]
    intrinsic = intrinsic[0]  # [S, 3, 3]
    extrinsic_homo = extrinsic_homo[0]  # extrinsic but shape [S,4,4]
    t_extrinsic = extrinsic[:, :3, 3]  # [B, S, 3]



def unproject_depth_map_to_point_map_index(
    depth_map: np.ndarray,  # shape: [S, H, W, 1]
    extrinsics_cam: np.ndarray,  # shape: [S, 3, 4]
    intrinsics_cam: np.ndarray,  # shape: [S, 3, 3]
    extrinsics_homo: np.ndarray,  # shape: [S, 4, 4]
    scale_factor: float,
) -> np.ndarray:
    '''Inference for 3D point cloud '''

    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.cpu().numpy()
        depth_map = depth_map * scale_factor
    if isinstance(extrinsics_cam, torch.Tensor):
        extrinsics_cam = extrinsics_cam.cpu().numpy()
        extrinsics_cam = torch.tensor(extrinsics_cam, dtype=torch.float32)
    if isinstance(intrinsics_cam, torch.Tensor):
        intrinsics_cam = intrinsics_cam.cpu().numpy()
    if isinstance(extrinsics_homo, torch.Tensor):
        extrinsics_homo = extrinsics_homo.cpu().numpy()

    world_points_list = []
    ref_inv = np.linalg.inv(extrinsics_homo[0])

    # convert extrinsic to cam to world
    cam_to_world_extrinsic = closed_form_inverse_se3(extrinsics_cam)

    R_cam_to_world = cam_to_world_extrinsic[:, :3, :3]
    t_cam_to_world = cam_to_world_extrinsic[:, :3, 3]
    t_cam_to_world = t_cam_to_world[:, :, None]
    t_scaled = t_cam_to_world.clone()
    t_scaled[:, : , 0] *= scale_factor  # scale forward/back translation only

    extrinsic_scaled = closed_form_inverse_se3(np.concatenate([R_cam_to_world, t_scaled], axis=2))

    for frame_idx in range(depth_map.shape[0]):
        cam_points, _, _ = depth_to_world_coords_points(
            depth_map[frame_idx].squeeze(-1),
            extrinsic_scaled[frame_idx],
            intrinsics_cam[0],
        )  # [H, W, 3]

        world_points_list.append(cam_points)

    world_points_array = np.stack(world_points_list, axis=0)  # [S, H, W, 3]
    return world_points_array, t_scaled



# -------------------------- Convert VGGT point map to SONATA format -----------------------------
def convert_vggt_to_sonata(
    point_map_by_unprojection: np.ndarray, images=not None, conf_threshold=math.inf
):
    ''''Convert VGGT point map to Sonata usable format'''
    def normal_from_cross_product(points_2d: np.ndarray) -> np.ndarray:
        dzdy = points_2d[1:, :-1, :] - points_2d[:-1, :-1, :]  # vertical diff
        dzdx = points_2d[:-1, 1:, :] - points_2d[:-1, :-1, :]  # horizontal diff
        normals = np.cross(dzdx, dzdy)
        norms = np.linalg.norm(normals, axis=-1, keepdims=True)
        normals = np.divide(
            normals, norms, out=np.zeros_like(normals), where=norms != 0
        )
        return normals  # [H-1, W-1, 3]

    S, H, W, _ = point_map_by_unprojection.shape  # S, H, W, 3
    H_valid = H - 1
    W_valid = W - 1
    coords_cropped = []
    colors_cropped = []
    normals_list = []

    for s in range(S):
        coords = point_map_by_unprojection[s, :H_valid, :W_valid].reshape(-1, 3)
        coords_cropped.append(coords)

        normals = normal_from_cross_product(
            point_map_by_unprojection[s]
        )  # [H-1, W-1, 3]
        normals_list.append(normals.reshape(-1, 3))  # [(H-1)*(W-1), 3]

        if images is not None:
            img = images[0, s] if images.dim() == 5 else images[s]
            img_np = img.permute(1, 2, 0).cpu().numpy()
            color = img_np[:H_valid, :W_valid].reshape(-1, 3)
            colors_cropped.append(color)

    coords_all = np.concatenate(coords_cropped, axis=0)
    normals_all = np.concatenate(normals_list, axis=0)
    colors_all = np.concatenate(colors_cropped, axis=0)

    # depth mask
    z_values = coords_all[:, 1]
    height_mask = z_values < conf_threshold

    coords_all = coords_all[height_mask]
    normals_all = normals_all[height_mask]
    colors_all = colors_all[height_mask]

    sonata_dict = {
        "coord": torch.from_numpy(coords_all).float(),
        "normal": torch.from_numpy(normals_all).float(),
        "color": torch.from_numpy(colors_all).float(),
    }

    return sonata_dict


# Take scale factor
path = os.path.abspath(os.path.join(os.path.dirname(__file__), "share_var.py"))
with open(path, "r") as f:
    for line in f.readlines():
        if "scale_factor" in line:
            sf = float(line.split("=")[1].strip())
print(f'Previous scale factor: {sf}\n')


# Call function to get point cloud
point_map_by_unprojection, t_extrinsic_scaled = unproject_depth_map_to_point_map_index(
    depth_map,
    extrinsic,
    intrinsic,
    extrinsic_homo,  # [S,4,4]
    scale_factor=sf,
)


torch.save(t_extrinsic_scaled, "t_extrinsic_scaled.pt")
print(f"t cam-to-world scaled extrinsics: {t_extrinsic_scaled}\n")  # uncomment if like to visualise
print(f"Translation part of extrinsic saved to: t_extrinsic_scaled.pt\n")


# Output
print(f"Using scale factor: {sf}")
sonata_data = convert_vggt_to_sonata(point_map_by_unprojection, images=images)

torch.save(sonata_data, "t_extrinsic_scaled.pt")
print(sonata_data.keys())
torch.save(sonata_data, "predictions.pt")
print("\nSonata formatted predictions saved to predictions.pt \n")
