import torch
from tqdm.auto import tqdm
import viser.transforms as viser_tf
import numpy as np


def cam_positions(data):
    extrinsic = data['extrinsic'].squeeze(0) #(S, 3, 4)

    S = extrinsic.shape[0]
    img_ids = range(S)

    print(f"Extrinsics shape: {extrinsic.shape}")
    coords = []
    for img_id in tqdm(img_ids):
        cam2world_3x4 = extrinsic[img_id].cpu().numpy()
        T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)
        T_world_camera = T_world_camera.wxyz_xyz[:4]  # Extract translation part
        print(f"Camera {img_id} T_world_camera: {T_world_camera}")

        coords.append(T_world_camera)

    torch.save(coords, 'cam_positions.pt')
    print("Camera positions saved to 'cam_positions.pt'")


if __name__ == "__main__":
    data = torch.load('vggt_raw_output.pt')
    print("Predict each cam frame position world coordinates:")
    cam_positions(data)

    

