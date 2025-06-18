import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import torch.linalg
from tqdm.auto import tqdm
import viser.transforms as viser_tf
from vggt.utils.geometry import closed_form_inverse_se3

# coordinates follow untransformed vggt axis, i.e. x right, y down, z forward
def cam_positions(data=torch.load('predictions.pt')): 
    
    extrinsics_cam = data["extrinsic"].squeeze(0)  # (S, 3, 4)
    
    cam_to_world_mat = closed_form_inverse_se3(extrinsics_cam)  # compute inverse, shape (S, 4, 4) typically
    cam_to_world = cam_to_world_mat[:, :3, :]

    S = extrinsics_cam.shape[0]
    xyz = np.zeros((S,3))
    wxyz = np.zeros((S,4))

    for img_id in tqdm(range(S)):
        cam2world_3x4 = cam_to_world[img_id].cpu().numpy()  # (3, 4)
        R_mat = cam2world_3x4[:, :3]  # rotation matrix (3x3)
        t_vec = cam2world_3x4[:, 3]   # translation vector (3,)

        # Convert rotation matrix to quaternion xyzw then back to wxyz
        quat_xyzw = R.from_matrix(R_mat).as_quat()
        quat_wxyz = np.roll(quat_xyzw, 1)

        wxyz[img_id] = quat_wxyz
        xyz[img_id] = t_vec
        #print(f"Camera {img_id} position (wxyz): {wxyz}")
    
    torch.save(wxyz, 'cam_rotation.pt')
    torch.save(xyz, 'cam_translation.pt')
    print("Camera roation saved to 'cam_rotation.pt', Camera translation saved to 'cam_translation.pt'")


# transform coordinates into sonata coord system
def transform(rotation=torch.load('cam_rotation.pt'), translation=torch.load('cam_translation.pt')):
    transformation = torch.tensor([
        [1,  0,  0],   # x right
        [0,  0, -1],   # z up
        [0, -1,  0],   # y towards
    ], dtype=torch.float32)

    for i in range(rotation.shape[0]):
        quat_wxyz = rotation[i]
        quat_xyzw = np.roll(quat_wxyz, -1)  # wxyz → xyzw

        rot = R.from_quat(quat_xyzw) #rotation matrix
        r_old = torch.from_numpy(rot.as_matrix()).float()
        r_new = transformation @ r_old

        U, _, Vt = torch.linalg.svd(r_new)
        R_ortho = U @ Vt
        if torch.det(R_ortho) < 0:
            U[:, -1] *= -1
            R_ortho = U @ Vt

        # Convert back to quaternion
        rot_new = R.from_matrix(R_ortho.numpy())
        quat_xyzw_new = rot_new.as_quat()
        quat_wxyz_new = np.roll(quat_xyzw_new, 1)  # xyzw → wxyz

        rotation[i] = torch.from_numpy(quat_wxyz_new)

    for i in range(translation.shape[0]):
        transformed_translation = transformation @ torch.tensor(translation[i], dtype=torch.float32)
        translation[i] = transformed_translation.numpy()

    torch.save(rotation, 'cam_rotation.pt')
    torch.save(translation, 'cam_translation.pt')
    print("Camera rotation saved to 'cam_rotation_sonata.pt', Camera translation saved to 'cam_translation_sonata.pt'")


if __name__ == "__main__":
    cam_positions()
    transform()

    

