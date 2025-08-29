import os

import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
from scipy.spatial import distance

import sonata

"""This script computes the scale factor based on the predicted point cloud and semantic segmentation results from sonata_inference.py."""


print(
    "\n################################# Computing Scale Factor ##################################\n"
)


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
    coords = torch.tensor(point["coord"]).cpu().detach().numpy()
    mask = np.array([name[i] == class_name for i in range(len(name))])
    return coords[mask]


# ------------- compute scale factor -------------
def scale_coord(frame_distance: float, min_depth) -> float:
    sf = frame_distance / min_depth
    return abs(sf)


if __name__ == "__main__":
    # Load predictions
    point = torch.load("sonata_points.pt")
    print(point.keys())

    name = torch.load("name.pt")
    print(f"Loaded names")

    frame_distance = 1.45  # modify if needed

    # get overall min floor coordinate compute scale factor
    floor_coords = get_coords_by_class(point, "floor", name)
    print(
        f"\nMax floor coords: {max(floor_coords[:, 1])}, min floor coords: {min(floor_coords[:, 1])}"
    )
    min_index = np.argmax(floor_coords[:, 1])
    min_coord = floor_coords[min_index]
    print(f"Original min floor coord: {min_coord}")
    print(f"Calibrated min floor depth: {min_coord[1] + frame_distance}")

    # Get scale factor
    path = os.path.join(os.path.dirname(__file__), "share_var.py")
    os.path.abspath(path)
    with open(os.path.abspath(path), "r") as f:
        content = f.read().strip()

    scale_factor = float(content.split("=")[1].strip())
    print("Old Scale factor:", scale_factor)

    scale_factor = scale_coord(frame_distance=frame_distance, min_depth=min_coord[1])
    print(f"Scaled by frame dis {frame_distance} m, Scale factor: {scale_factor}\n")

    # Save updated scale factor to share_var.py
    with open(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "share_var.py")),
        "w",
    ) as f:
        f.write(f"scale_factor = {scale_factor}\n")

    torch.cuda.empty_cache()
