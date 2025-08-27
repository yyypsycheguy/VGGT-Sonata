import os

import numpy as np
from scipy.spatial import distance
import open3d as o3d
import torch
import torch.nn as nn

import sonata

print("################################# Scaling ##################################\n")

# ------------ get coords of selected classes ------------
def get_coords_by_class(point, class_name, name):
    """
    Get coordinates of points belonging to a specific class.
    Args:
        point (dict): Dictionary containing point cloud data.
        class_name (str): Name of the class to filter by.
    Returns:
        np.ndarray: Coordinates of points belonging to the specified class.
    """
    coords = point["coord"].cpu().detach().numpy()
    mask = np.array([name[i] == class_name for i in range(len(name))])
    return coords[mask]

# ------------- compute scale factor -------------
def scale_coord(frame_dis: float, min_depth) -> float:
    sf = frame_dis / min_depth / 1000
    return sf


if __name__ == "__main__":

    # Load predictions
    point = torch.load("results.pt")
    print(point.keys())

    name = torch.load("name.pt")
    print(f"Loaded names")

    frame_dis = 1.45  # modify if needed

    # get overall min floor coordinate compute scale factor
    floor_coords = get_coords_by_class(point, "floor", name)
    print(f"\nMax floor coords: {max(floor_coords[:, 2])}, min floor coords: {min(floor_coords[:, 2])}")
    min_index = np.argmin(floor_coords[:, 2])
    min_coord = floor_coords[min_index]
    print(f"Original min floor coord: {min_coord}")
    print(f"Calibrated min floor depth: {min_coord[2] + frame_dis}")


    path = os.path.join(os.path.dirname(__file__), "../vggt/share_var.py")
    os.path.abspath(path)
    with open(os.path.abspath(path), "r") as f:
        content = f.read().strip()

    scale_factor = float(content.split("=")[1].strip())
    print("Old Scale factor:", scale_factor)

    scale_factor = scale_coord(frame_dis=frame_dis, min_depth=min_coord[2])
    print(f"Scaled by frame dis {frame_dis} m, Scale factor: {scale_factor}\n")

    # Save updated scale factor to share_var.py
    with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "../vggt/share_var.py")),"w",) as f:
        f.write(f"scale_factor = {scale_factor}\n")

    torch.cuda.empty_cache()
