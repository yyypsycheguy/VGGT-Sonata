import numpy as np
import torch
from plyfile import PlyData, PlyElement

# Load VGGT output
vggt_output = torch.load("predictions.pt")  # Your .pt file
world_points = vggt_output["world_points"]  # Shape: [B, S, H, W, 3]

# Assuming batch_size=1 and using the first frame (S=0)
points = world_points[0, 0].reshape(-1, 3).cpu().numpy()  # Shape: [H*W, 3]

# Optional: Filter low-confidence points (using world_points_conf)
conf = vggt_output["world_points_conf"][0, 0].reshape(-1).cpu().numpy()
points = points[conf > 0.5]  # Keep points with confidence > 50%

# Save as .ply
vertex = np.array(
    [(x, y, z) for (x, y, z) in points], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")]
)
PlyData([PlyElement.describe(vertex, "vertex")]).write("predictions.ply")
