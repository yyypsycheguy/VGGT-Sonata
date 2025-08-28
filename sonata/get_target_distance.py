import os

from compute_scale_factor import get_coords_by_class

import numpy as np
from scipy.spatial import distance
import open3d as o3d
import torch
import torch.nn as nn

import sonata

print("################################# Get target distance ##################################\n")

if __name__ == "__main__":

    # Load predictions
    point = torch.load("results.pt")
    print(point.keys())

    name = torch.load("name.pt")
    print(f"Loaded names")

    # get coords of target class
    frame_dis = 1.45  # modify if needed
    target = "chair" # change to your target object name
    target_coords = get_coords_by_class(point, target, name)


    print(f"\nMax {target} coords: {max(target_coords[:, 2])}, min chair coords: {min(target_coords[:, 2])}")
    max_index = np.argmax(target_coords[:, 2])
    max_coord = target_coords[max_index]
    print(f"Original max {target} coord: {max_coord}")
    target_calibrated_dis = max_coord[2] + frame_dis
    print(f"Calibrated max {target} depth: {target_calibrated_dis}")
    max_coord[1] = target_calibrated_dis
    xy_target = max_coord[:2]
    print(f"Calibrated {target} coord xyz coordinate: {xy_target}")

    target_right_dis = max_coord[1]

        
    # Calculate manhattan distance
    # extrinsic = torch.load("t_extrinsic_scaled.pt")
    # for img in extrinsic:
    #     if xy_target[0] - img[0] < 0.1:
    #         print(f"Found matching frame with x coord: {img[0]}, y coord: {img[1]}")
    #         break
    # manhattan_dist = distance.cityblock(xy_target, np.array([0, 0, 0]))


    # Write distance ouput to dis_output.py
    output_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "dis_output.py")
    )
    print("Writing to:", output_path)
    with open(output_path, "w") as f:
        f.write(f"dis_y = {target_calibrated_dis}\n")
        f.write(f"dis_x = {target_right_dis}\n")
    print(f"Updated distance: forward:{target_calibrated_dis}, and sideways: {target_right_dis} saved to dis_output.py \n")
