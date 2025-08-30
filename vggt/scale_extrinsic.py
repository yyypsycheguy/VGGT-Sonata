import os

import numpy as np
import torch
from vggt.utils.geometry import closed_form_inverse_se3

''' This script scales extrinsics by the computed scale factor'''

print("\n############################## Scale Extrinsic ######################################")

def extrinsic_scaling(
    extrinsics_cam: np.ndarray,  # shape: [S, 3, 4]
    scale_factor: float,
) -> np.ndarray:
    
    '''Scale extrinsics. Would be original extrinsic if scale factor = 1.0
    
    args:
        extrinsics_cam: complete extrinsic from camera to world, shape [S, 3, 4]
        scale_factor: scale factor to scale extrinsics
    return:
        scaled translation vector from extrinsic, shape [S, 3, 1]'''

    if isinstance(extrinsics_cam, torch.Tensor):
        extrinsics_cam = extrinsics_cam.cpu().numpy()
        extrinsics_cam = torch.tensor(extrinsics_cam, dtype=torch.float32)

    # convert extrinsic to cam to world
    cam_to_world_extrinsic = closed_form_inverse_se3(extrinsics_cam)

    t_cam_to_world = cam_to_world_extrinsic[:, :3, 3]
    t_cam_to_world = t_cam_to_world[:, :, None]
    t_scaled = t_cam_to_world.clone()
    t_scaled[:, : , 0] *= scale_factor

    return t_scaled


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Take scale factor
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../sonata/share_var.py"))
    with open(path, "r") as f:
        for line in f.readlines():
            if "scale_factor" in line:
                sf = float(line.split("=")[1].strip())
    print(f'Previous scale factor: {sf}\n')

    # Scale extrinsics
    extrinsics = torch.load("extrinsic.pt")
    scaled_extrinsics = extrinsic_scaling(extrinsics, sf)
    scaled_extrinsics = torch.tensor(scaled_extrinsics, dtype=torch.float32)
    print(f"Extrinsics: {scaled_extrinsics}")

    torch.save(scaled_extrinsics, "extrinsic_scaled.pt")
    print(f"Scaled extrinsics by factor {sf}, saved to extrinsic_scaled.pt\n")