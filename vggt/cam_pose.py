import numpy as np
import torch
from tqdm.auto import tqdm
import viser.transforms as viser_tf
from vggt.utils.geometry import closed_form_inverse_se3


def cam_positions(data):
    
    extrinsics_cam = data["extrinsic"].squeeze(0)  # (S, 3, 4)
    print(f"Extrinsics shape: {extrinsics_cam.shape}")
    
    cam_to_world_mat = closed_form_inverse_se3(extrinsics_cam)  # compute inverse, shape (S, 4, 4) typically
    print(f"Camera to world matrix shape: {cam_to_world_mat.shape}")
    #store only (3,4) portion
    cam_to_world = cam_to_world_mat[:, :3, :]
    print(f"Camera to world matrix (3x4) shape: {cam_to_world.shape}")

    S = extrinsics_cam.shape[0]
    img_ids = range(S)
    xyz = np.zeros((S,3))
    wxyz = np.zeros((S,4))

    for img_id in tqdm(img_ids):
        cam2world_3x4 = cam_to_world[img_id].cpu().numpy()   # cam_to_world = extrinsic, shape (S, 3, 4)
        T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)
        wxyz[img_id] = T_world_camera.rotation().wxyz # type np array
        xyz[img_id] = T_world_camera.translation()
        #print(f"Camera {img_id} position (wxyz): {wxyz}")
        #print(f"Camera {img_id} T_world_camera: {T_world_camera}")
    
    torch.save(wxyz, 'cam_rotation.pt')
    torch.save(xyz, 'cam_translation.pt')
    print("Camera roation saved to 'cam_rotation.pt', Camera translation saved to 'cam_translation.pt'")


if __name__ == "__main__":
    data = torch.load('predictions.pt')
    print("Predict each cam frame position world coordinates:")
    print(cam_positions(data))

    print(f"cam rotation: {torch.load('cam_rotation.pt')}")
    print(f"cam translation: {torch.load('cam_translation.pt')}")

    

