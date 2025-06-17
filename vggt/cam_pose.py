import numpy as np
import torch
from tqdm.auto import tqdm
import viser.transforms as viser_tf
from vggt.utils.geometry import closed_form_inverse_se3


def cam_positions(data):
    points = data['world_points'].reshape(-1,3)  # flatten to Nx3 (N, 3)
    extrinsics_cam = data["extrinsic"].squeeze(0)  # (S, 3, 4)
    print(f"Extrinsic:{extrinsics_cam}")
    print(f"Extrinsics shape: {extrinsics_cam.shape}")
    
    cam_to_world_mat = closed_form_inverse_se3(extrinsics_cam)  # compute inverse, shape (S, 4, 4) typically
    print(f"Camera to world matrix shape: {cam_to_world_mat.shape}")
    # For convenience, we store only (3,4) portion
    cam_to_world = cam_to_world_mat[:, :3, :]
    print(f"Camera to world matrix (3x4) shape: {cam_to_world.shape}")

    S = extrinsics_cam.shape[0]
    img_ids = range(S)

    for img_id in tqdm(img_ids):
        cam2world_3x4 = cam_to_world[img_id].cpu().numpy()   # cam_to_world = extrinsic, shape (S, 3, 4)
        T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)
        print(f"Camera {img_id} T_world_camera: {T_world_camera}")
    
    torch.save(T_world_camera, 'cam_positions.pt')
    print("Camera positions saved to 'cam_positions.pt'")


if __name__ == "__main__":
    data = torch.load('vggt_raw_output.pt')
    print("Predict each cam frame position world coordinates:")
    print(cam_positions(data))

    

