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
        # Predict attributes including cameras, depth maps, and point maps.
        predictions = model(images)

# get extrinsic
extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
predictions["extrinsic"] = extrinsic
print(f"Extrinsic in main(): {extrinsic}")
predictions["intrinsic"] = intrinsic


'''def convert_vggt_to_sonata(point_map_by_unprojection, images, scale_factor=10.0, confidence_threshold=0.01):
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
print("Sonata formatted predictions saved to predictions.pt")'''
