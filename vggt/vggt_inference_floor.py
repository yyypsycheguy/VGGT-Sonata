import os

import numpy as np
import math
import torch

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.geometry import depth_to_world_coords_points
from vggt.utils.geometry import closed_form_inverse_se3

# only uncomment when using reachy
'''scale_factor = float(input('scale factor (meters):'))
cam_frame_dis = float(input('Distance from camera to edge of floor (meters):'))
if scale_factor <= 0 or cam_frame_dis <= 0:
    raise ValueError("Scale factor and camera frame distance must be positive numbers.")
if scale_factor < cam_frame_dis:
    raise ValueError("Scale factor must be greater than camera frame distance.")
if isinstance(scale_factor or cam_frame_dis, (int, float)) is False:
    raise TypeError("Scale factor and camera frame distance must be of type float.")'''

'''with open("share_var.py", "w") as f:
    f.write(f"cam_frame_dis = {cam_frame_dis}")'''

    
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

model.eval()

# Load and preprocess example images (replace with your own image paths)
image_names = []
for img in os.listdir("images"):
    if img.endswith((".jpg", ".jpeg", ".png")):
        image_names.append(os.path.join("images", img))

images = load_and_preprocess_images(image_names).to(device)


with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        images = images[None]  # add batch dimension
        aggregated_tokens_list, ps_idx = model.aggregator(images)
                
    # Predict Cameras
    pose_enc = model.camera_head(aggregated_tokens_list)[-1]
 
    # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:]) # (B, S, 3, 4) and (B, S, 3, 3)

    B, V = extrinsic.shape[:2]  # [1, 6, 3, 4]
    extrinsic_homo = torch.eye(4, device=device).repeat(B, V, 1, 1)  # [1, 6, 4, 4]
    extrinsic_homo[:, :, :3, :] = extrinsic
    transformation = torch.eye(4, device=device)         
    transformation = torch.tensor([
    [1,  0,  0, 0],  # x right
    [0,  0,  -1, 0],  # z up
    [0,  -1, 0, 0],  # y towards
    [0,  0,  0, 1],
    ], dtype=torch.float32, device=extrinsic.device) 
    
    transformation = transformation[None, None, :, :]  # [1, 1, 4, 4]
    extrinsic_homo = (extrinsic_homo @ transformation) # [B, S, 4, 4]
    extrinsic = extrinsic_homo[:, :,:3,:]

    # Predict Depth Maps
    depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
    
    # remove batch dimension
    depth_map = depth_map[0]  # [S, H, W, 1]
    extrinsic = extrinsic[0]  # [S, 3, 4]
    intrinsic = intrinsic[0]  # [S, 3, 3]
    extrinsic_homo = extrinsic_homo[0]  # [S, 4, 4]

    vggt_raw_output = unproject_depth_map_to_point_map(
        depth_map,
        extrinsic,
        intrinsic
    )


def unproject_depth_map_to_point_map_index(
    depth_map: np.ndarray,         # shape: [S, H, W, 1]
    extrinsics_cam: np.ndarray,    # shape: [S, 3, 4]
    intrinsics_cam: np.ndarray,    # shape: [S, 3, 3]
    extrinsics_homo: np.ndarray,   # shape: [S, 4, 4]
    scale_factor: float,
    )-> np.ndarray:

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

    # Scale translation matrix in extrinsic
    cam_to_world_extrinsic = closed_form_inverse_se3(extrinsics_cam)

    R_cam_to_world = cam_to_world_extrinsic[:, :3, :3]
    t_cam_to_world = cam_to_world_extrinsic[:, :3, 3] 
    t_cam_to_world = t_cam_to_world[:,:,None]
    t_scaled = t_cam_to_world * scale_factor
    extrinsic_scaled = closed_form_inverse_se3(np.concatenate([R_cam_to_world, t_scaled], axis=2))

    for frame_idx in range(depth_map.shape[0]):

        cam_points, _, _ = depth_to_world_coords_points(
            depth_map[frame_idx].squeeze(-1),
            extrinsic_scaled[frame_idx], 
            intrinsics_cam[0]
        )  # [H, W, 3]

        H, W, _ = cam_points.shape
        cam_points_flat = cam_points.reshape(-1, 3)
        ones = np.ones((cam_points_flat.shape[0], 1))
        cam_points_h = np.concatenate([cam_points_flat, ones], axis=1)  # [N, 4]

        transform = ref_inv @ extrinsics_homo[frame_idx]
        world_points_flat = (transform @ cam_points_h.T).T[:, :3]  # [N, 3]
        world_points = world_points_flat.reshape(H, W, 3)

        world_points_list.append(world_points)

    world_points_array = np.stack(world_points_list, axis=0)  # [S, H, W, 3]
    return world_points_array


path = os.path.abspath(os.path.join(os.path.dirname(__file__), "share_var.py"))
with open(path, 'r') as f:
    for line in f.readlines():
        if 'scale_factor' in line:
            sf = float(line.split('=')[1].strip())


point_map_by_unprojection = unproject_depth_map_to_point_map_index(
    depth_map,
    extrinsic,
    intrinsic,
    extrinsic_homo, # [S,4,4]
    scale_factor= sf
)

open(path, "w").close() # clears sf content

# -------------------------- Convert VGGT point map to SONATA format -----------------------------
def convert_vggt_to_sonata(point_map_by_unprojection: np.ndarray, images= not None, conf_threshold= math.inf):

    def normal_from_cross_product(points_2d: np.ndarray) -> np.ndarray:
        dzdy = points_2d[1:, :-1, :] - points_2d[:-1, :-1, :]  # vertical diff
        dzdx = points_2d[:-1, 1:, :] - points_2d[:-1, :-1, :]  # horizontal diff
        normals = np.cross(dzdx, dzdy)
        norms = np.linalg.norm(normals, axis=-1, keepdims=True)
        normals = np.divide(normals, norms, out=np.zeros_like(normals), where=norms != 0)
        return normals  # [H-1, W-1, 3]

    S, H, W, _ = point_map_by_unprojection.shape # S, H, W, 3
    H_valid = H - 1
    W_valid = W - 1
    coords_cropped = []
    colors_cropped = []
    normals_list = []

    for s in range(S):
        coords = point_map_by_unprojection[s, :H_valid, :W_valid].reshape(-1, 3)
        coords_cropped.append(coords)

        normals = normal_from_cross_product(point_map_by_unprojection[s])  # [H-1, W-1, 3]
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
        "color": torch.from_numpy(colors_all).float()
    }

    return sonata_dict


if sf == 1.0:
    sonata_data = convert_vggt_to_sonata(vggt_raw_output, images=images)
else:
    print(f"Using scale factor: {sf}")
    sonata_data = convert_vggt_to_sonata(point_map_by_unprojection, images=images)

for key, value in sonata_data.items():
    if isinstance(value, (torch.Tensor, np.ndarray)):
        print(f"{key}: shape = {value.shape}")
#torch.save(sonata_data, "predictions.pt")
torch.save(sonata_data, "predictions.pt")

print(sonata_data.keys())
print("Sonata formatted predictions saved to predictions.pt \n")

