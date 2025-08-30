import os

import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
from compute_scale_factor import get_coords_by_class
from scipy.spatial import distance

import sonata

print(
    "################################# Get target distance ##################################\n"
)

if __name__ == "__main__":
    # Load predictions
    point = torch.load("scaled_sonata_points.pt")
    print(point.keys())

    name = torch.load("name.pt")
    print(f"Loaded names")

    # get coords of target class
    target = "chair"  # change to your target object name
    target_coords = get_coords_by_class(
        point, target, name
    )  # [n,3] x right,y towards,z up

    print(
        f"\nMax {target} coords: {max(target_coords[:, 1])}, min chair coords: {min(target_coords[:, 1])}"
    )
    max_index = np.argmax(abs(target_coords[:, 1]))
    max_coord = target_coords[max_index]
    print(f"Original max {target} coord: {max_coord}")

    target_forward_dis = max_coord[1] 
    xy_target = max_coord[:2]
    print(f"Calibrated {target} coord xyz coordinate: {xy_target}")

    target_right_dis = max_coord[0]

    # Write distance ouput to dis_output.py
    output_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "dis_output.py")
    )
    print("Writing to:", output_path)
    with open(output_path, "w") as f:
        f.write(f"dis_y = {target_forward_dis}\n")
        f.write(f"dis_x = {target_right_dis}\n")
    print(
        f"Updated distance: forward:{target_forward_dis}, and sideways: {target_right_dis} saved to dis_output.py \n"
    )
