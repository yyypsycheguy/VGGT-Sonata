import os

import numpy as np
import torch

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images



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
    with torch.amp.autocast("cuda", dtype=dtype):
        # Predict attributes including cameras, depth maps, and point maps.
        predictions = model(images)
print(f"prediction keys:{predictions.keys()}")
def convert_vggt_to_sonata(predictions, scale_factor=3.0, confidence_threshold=0.01):
    import numpy as np
    import torch

    def normal_from_cross_product(points_2d: np.ndarray) -> np.ndarray:
        xyz_points_pad = np.pad(points_2d, ((0, 1), (0, 1), (0, 0)), mode="symmetric")
        xyz_points_ver = (xyz_points_pad[:, :-1, :] - xyz_points_pad[:, 1:, :])[:-1, :, :]
        xyz_points_hor = (xyz_points_pad[:-1, :, :] - xyz_points_pad[1:, :, :])[:, :-1, :]
        xyz_normal = np.cross(xyz_points_hor, xyz_points_ver)
        xyz_dist = np.linalg.norm(xyz_normal, axis=-1, keepdims=True)
        xyz_normal = np.divide(
            xyz_normal, xyz_dist, out=np.zeros_like(xyz_normal), where=xyz_dist != 0
        )
        return xyz_normal

    world_points = predictions["world_points"] #[B, S, H, W, 3] 
    grid_points = world_points[0, 0].cpu().numpy()
    coords = grid_points.reshape(-1, 3) * (-scale_factor) # Flatten to Nx3

    # Compute normals (same shape as input)
    normals_grid = normal_from_cross_product(grid_points)  
    
    # Flatten normals
    normals_flat = normals_grid.reshape(-1, 3)
    
    # Confidence mask
    conf = predictions["world_points_conf"][0, 0].cpu().numpy()  
    conf_flat = conf.reshape(-1)  
    valid_mask = conf_flat > confidence_threshold

    # Apply mask to both coords and normals
    coords = coords[valid_mask]
    normals_flat = normals_flat[valid_mask]


    sonata_dict = {"coord": coords, "normal": normals_flat}

    # Optional: include RGB color (colors don't need scaling)
    if "images" in predictions:
        img = predictions["images"][0, 0].permute(1, 2, 0).cpu().numpy() 
        img_flat = img.reshape(-1, 3)
        sonata_dict["color"] = img_flat[valid_mask]

    # Convert all outputs to torch tensors
    for k, v in sonata_dict.items():
        if isinstance(v, np.ndarray):
            sonata_dict[k] = torch.from_numpy(v).float()

    return sonata_dict


# Usage example:
sonata_data = convert_vggt_to_sonata(predictions)
for key, value in sonata_data.items():
    if isinstance(value, (torch.Tensor, np.ndarray)):
        print(f"{key}: shape = {value.shape}\n")
torch.save(sonata_data, "predictions.pt")



print(sonata_data.keys())
print("Sonata formatted predictions saved to predictions.pt")
