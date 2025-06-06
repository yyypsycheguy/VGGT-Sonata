import os

import numpy as np
import torch

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map


device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
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
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

    B, V = extrinsic.shape[:2]  # [1, 6, 3, 4]
    extrinsic_homo = torch.eye(4, device=device).repeat(B, V, 1, 1)  # [1, 6, 4, 4]
    extrinsic_homo[:, :, :3, :] = extrinsic

    transformation = torch.eye(4, device=device)         
    transformation = torch.tensor([
    [1,  0,  0, 0],  # x 
    [0,  0,  -1, 0],  # z
    [0,  -1, 0, 0],  # y
    [0,  0,  0, 1],
], dtype=torch.float32, device=extrinsic.device) 
    transformation = transformation[None, None, :, :]  # [1, 1, 4, 4]
    extrinsic = extrinsic_homo @ transformation
    extrinsic = extrinsic[:,:,:3,:]

    # Predict Depth Maps
    depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    # Predict Point Maps
    point_map, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx)
        
    # Construct 3D Points from Depth Maps and Cameras
    # which usually leads to more accurate 3D points than point map branch
    point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map.squeeze(0), 
                                                                extrinsic.squeeze(0), 
                                                                intrinsic.squeeze(0))

    # Predict Tracks
    # choose your own points to track, with shape (N, 2) for one scene
    query_points = torch.FloatTensor([[100.0, 200.0], 
                                        [60.72, 259.94]]).to(device)
    track_list, vis_score, conf_score = model.track_head(aggregated_tokens_list, images, ps_idx, query_points=query_points[None])


def convert_vggt_to_sonata(point_map_by_unprojection, images, scale_factor=10.0, confidence_threshold=0.01):
    import numpy as np
    import torch

    def normal_from_cross_product(points_2d: np.ndarray) -> np.ndarray:
        dzdy = points_2d[1:, :-1, :] - points_2d[:-1, :-1, :]  # vertical diff
        dzdx = points_2d[:-1, 1:, :] - points_2d[:-1, :-1, :]  # horizontal diff
        normals = np.cross(dzdx, dzdy)
        norms = np.linalg.norm(normals, axis=-1, keepdims=True)
        normals = np.divide(normals, norms, out=np.zeros_like(normals), where=norms != 0)
        return normals  # [H-1, W-1, 3]

    S, H, W, _ = point_map_by_unprojection.shape
    H_valid = H - 1
    W_valid = W - 1
    coords_cropped = []
    colors_cropped = []
    normals_list = []

    for s in range(S):
        coords = point_map_by_unprojection[s, :H_valid, :W_valid].reshape(-1, 3) * (scale_factor)
        coords_cropped.append(coords)

        normals = normal_from_cross_product(point_map_by_unprojection)  # [H-1, W-1, 3]
        normals_list.append(normals.reshape(-1, 3))     # [(H-1)*(W-1), 3]

        if images is not None:
            img = images[0, s] if images.dim() == 5 else images[s]
            img_np = img.permute(1, 2, 0).cpu().numpy()
            color = img_np[:H_valid, :W_valid].reshape(-1, 3)
            colors_cropped.append(color)

    coords_all = np.concatenate(coords_cropped, axis=0)
    normals_all = np.concatenate(normals_list, axis=0)
    colors_all = np.concatenate(colors_cropped, axis=0)

    sonata_dict = {
    "coord": coords_all,
    "normal": normals_all,
    "color": colors_all
    }

    # Convert all to torch
    for k, v in sonata_dict.items():
        sonata_dict[k] = torch.from_numpy(v).float()

    return sonata_dict




sonata_data = convert_vggt_to_sonata(point_map_by_unprojection, images=images)
for key, value in sonata_data.items():
    if isinstance(value, (torch.Tensor, np.ndarray)):
        print(f"{key}: shape = {value.shape}\n")
torch.save(sonata_data, "predictions.pt")



print(sonata_data.keys())
print("Sonata formatted predictions saved to predictions.pt")
