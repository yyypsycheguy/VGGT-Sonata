import os
import torch
import numpy as np
from vggt.utils.load_fn import load_and_preprocess_images
from vggt_inference_floor import convert_vggt_to_sonata

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_names = []
for img in os.listdir("images"):
    if img.endswith((".jpg", ".jpeg", ".png")):
        image_names.append(os.path.join("images", img))

images = load_and_preprocess_images(image_names).to(device)

def relocate_scene(image, data=torch.load("vggt_raw_output.pt")):
    world_points = data["world_points"]  # [S, H, W, 3]
    S, H, W, *_ = world_points.shape  # Fixed: underscore instead of asterisk
    
    world_points = world_points[image]  # Select the specific image's world points [H, W, 3]
    
    # Convert to tensor
    if isinstance(world_points, np.ndarray):
        world_points = torch.from_numpy(world_points)

    # Add batch dimension: (H, W, 3) -> (1, H, W, 3)  
    world_points = world_points.unsqueeze(0) 
    
    print(f"Processed image {image + 1}/{S}, Point cloud shape: {world_points.shape}")
    return world_points

def get_coord(world_points):
    """Convert one single selected image vggt world points to SONATA coordinates."""

    coords = convert_vggt_to_sonata(world_points, images=images)
    coords = coords["coord"].cpu().numpy()

    print(f"Converted coordinates shape: {coords.shape}")
    torch.save(coords, "vggt_relocated_coords.pt")
    print(f"Relocated coordinates saved to vggt_relocated_coords.pt")
    print(f"Relocates coordinates: {coords}, shape: {coords.shape}")
    # coords should now be [H, W, 3] in SONATA format, removed dimension of "batch"
    return coords



if __name__ == "__main__":
    image_index = 0  # sample image index <= S-1
    world_points = relocate_scene(image_index)
    coords = get_coord(world_points)