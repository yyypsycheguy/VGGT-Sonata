import os

import numpy as np
import torch
from compute_scale_factor import get_coords_by_class

"""'This script scales the point cloud by the computed scale factor from sonata/scaling.py"""

print("\n############################## Scale point cloud ######################################")

def scale_pointcloud(point, scale_factor):
    """Scale point cloud by scale factor"""
    point["coord"] = point["coord"] * scale_factor

    return point


if __name__ == "__main__":
    # Take scale factor
    path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../sonata/share_var.py")
    )
    with open(path, "r") as f:
        for line in f.readlines():
            if "scale_factor" in line:
                sf = float(line.split("=")[1].strip())
    print(f"Previous scale factor: {sf}\n")

    # Scale point cloud
    point = torch.load("sonata_points.pt")
    scaled_pointcloud = scale_pointcloud(point, sf)
    
    print(f"Point cloud coordinate range SCALE FLOOR COLOR:")
    print(f"x: {point['coord'][:, 0].min()} to {point['coord'][:, 0].max()}")
    print(f"y: {point['coord'][:, 1].min()} to {point['coord'][:, 1].max()}")
    print(f"z: {point['coord'][:, 2].min()} to {point['coord'][:, 2].max()}")

    torch.save(scaled_pointcloud, "scaled_sonata_points.pt")
    print(f"Scaled point cloud by factor {sf}, saved to scaled_sonata_points.pt")

    # Get floor coordinates for 2D map
    name = torch.load("name.pt")
    floor_coords = get_coords_by_class(scaled_pointcloud, "floor", name)

    torch.save(floor_coords, "scaled_floor_coords.pt")
    print("Saved scaled floor coordinates to floor_coords.pt\n")
