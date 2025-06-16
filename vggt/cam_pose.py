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
    rotations_dict = {}
    translations_dict = {}

    for img_id in tqdm(img_ids):
        cam2world_3x4 = extrinsic[img_id].cpu().numpy()
        T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)
        
        # Extract rotation (wxyz) and translation (xyz)
        rotation = T_world_camera.wxyz_xyz[:4]     # First 4 elements
        translation = T_world_camera.wxyz_xyz[4:]  # Last 3 elements

        # Store in dicts
        rotations_dict[f"Camera_{img_id}"] = rotation
        translations_dict[f"Camera_{img_id}"] = translation

        print(f"Camera {img_id} rotation (wxyz): {rotation}")
        print(f"Camera {img_id} translation (xyz): {translation}\n")

        #print(f"Camera {img_id} T_world_camera: {T_world_camera}")
        #coords.append(T_world_camera)

    torch.save({'rotations': rotations_dict, 'translations': translations_dict}, 'cam_positions.pt')
    print("Camera positions saved to 'cam_positions.pt'")


if __name__ == "__main__":
    data = torch.load('vggt_raw_output.pt')
    print("Predict each cam frame position world coordinates:")
    cam_positions(data)

    

