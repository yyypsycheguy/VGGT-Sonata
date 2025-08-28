import os

import numpy as np
import torch

from vggt.utils.geometry import closed_form_inverse_se3

''''This script scales the point cloud and extrinsics by the computed scale factor from sonata/scaling.py'''

print("\n############################## Apply Scale Factor to point cloud & extrinsic ######################################")


def extrinsic_scaling(
    extrinsics_cam: np.ndarray,  # shape: [S, 3, 4]
    scale_factor: float,
) -> np.ndarray:
    '''Scale extrinsics. Would be original extrinsic if scale factor = 1.0'''

    if isinstance(extrinsics_cam, torch.Tensor):
        extrinsics_cam = extrinsics_cam.cpu().numpy()
        extrinsics_cam = torch.tensor(extrinsics_cam, dtype=torch.float32)

    # convert extrinsic to cam to world
    cam_to_world_extrinsic = closed_form_inverse_se3(extrinsics_cam)

    t_cam_to_world = cam_to_world_extrinsic[:, :3, 3]
    t_cam_to_world = t_cam_to_world[:, :, None]
    t_scaled = t_cam_to_world.clone()
    t_scaled[:, : , 0] *= scale_factor  # scale forward/back translation only

    return t_scaled


def scale_pointcloud(point, scale_factor):
    '''Scale point cloud by scale factor'''
    point["coord"] = point["coord"] * scale_factor

    return point



if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Take scale factor
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../sonata/share_var.py"))
    with open(path, "r") as f:
        for line in f.readlines():
            if "scale_factor" in line:
                sf = float(line.split("=")[1].strip())
    print(f'Previous scale factor: {sf}\n')

    # Scale point cloud
    point = torch.load("../vggt/predictions.pt")
    scaled_pred = scale_pointcloud(point, sf)
    torch.save(scaled_pred, "predictions.pt")
    print(f"Scaled point cloud by factor {sf}, saved to predictions.pt")

    # Scale extrinsics
    extrinsics = torch.load("../vggt/extrinsic.pt")
    scaled_extrinsics = extrinsic_scaling(extrinsics, sf)
    print(f"Extrinsics: {scaled_extrinsics}")
    torch.save(scaled_extrinsics, "extrinsic.pt")
    print(f"Scaled extrinsics by factor {sf}, saved to extrinsic.pt\n")