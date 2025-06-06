import numpy as np
import open3d as o3d
import torch

data = torch.load("predictions.pt")
print(data.keys())
print(data["coord"])
pcd = o3d.io.read_point_cloud("predictions/predictions.ply")

# Calculate current height
points = np.asarray(pcd.points)
print("Current points shape:", points.shape)
print(points[:5])
current_height = np.max(points[:, 0]) - np.min(points[:, 0])
print("Current height:", current_height)

target_height = 3.0
scale_factor = target_height / current_height
print("Scale factor:", scale_factor)

# Apply scaling
scaled = pcd.scale(scale_factor, center=(0, 0, 0))
print("Scaled points shape:", np.asarray(scaled.points).shape)
array = np.asarray(scaled.points)
print(array[:5])
