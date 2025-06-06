import os
import numpy as np
import torch

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


def save_ply(points, filename):
    """Save Nx3 point cloud to .ply"""
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")


device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
model.eval()

# Load images
image_names = [os.path.join("images", f) for f in os.listdir("images")
               if f.lower().endswith((".jpg", ".jpeg", ".png"))]

images = load_and_preprocess_images(image_names).to(device)

with torch.no_grad():
    with torch.amp.autocast("cuda", dtype=dtype):
        predictions = model(images)

# === Extract and Save 3D points ===
# world_points: [B, S, H, W, 3]
world_points = predictions["world_points"]  # [B, S, H, W, 3]
world_points = world_points.detach().cpu().float()

B, S, H, W, _ = world_points.shape
output_dir = "predictions"
os.makedirs(output_dir, exist_ok=True)

for b in range(B):
    for s in range(S):
        pts = world_points[b, s].reshape(-1, 3)  # [H*W, 3]
        ply_path = os.path.join(output_dir, f"{os.path.basename(image_names[b])}_frame{s}.ply")
        save_ply(pts.numpy(), ply_path)
        print(f"Saved: {ply_path}")
