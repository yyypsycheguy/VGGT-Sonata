from vggt.utils.pose_enc import extri_intri_to_pose_encoding
from vggt.utils.load_fn import load_and_preprocess_images
import torch
import os


device = "cuda" if torch.cuda.is_available() else "cpu"

# Process images
image_names = []
for img in os.listdir("images"):
    if img.endswith((".jpg", ".jpeg", ".png")):
        image_names.append(os.path.join("images", img))

images = load_and_preprocess_images(image_names).to(device)


def cam_pose_finder(images, data=torch.load("vggt_raw_output.pt")):

    '''Returns:
    torch.Tensor: Encoded camera pose parameters with shape BxSx9.
        With "absT_quaR_FoV" type, the 9 dimensions are:
        - [:3] = absolute translation vector T (3D) camera's position in world coordinates
        - [3:7] = rotation as quaternion quat (4D) representing the camera's orientation
        - [7:] = field of view (2D) representing the camera's horizontal and vertical FoV angles '''
    
    extrinsic = data["extrinsic"]  # [B, S, 3, 4]
    intrinsic = data["intrinsic"]  # [B, S, 3, 3]
    image_size_hw = images.shape[-2:]
    cam_pose = extri_intri_to_pose_encoding(
        extrinsic, 
        intrinsic, 
        image_size_hw, 
    )

    return cam_pose


total_img = len(image_names)

def pose_to_coords(pose, img_index):
    '''
    args: pose  = position encoding, output of cam_pose_finder
          img_index = image index in the series
    Returns: coordinates realtive to the origin (first image in the series).
    Convert camera pose to 3D coordinates.'''

    cam_position = pose[:, :3]  # Bx3 (x,y,z)
    if img_index > total_img-1 or img_index < 0:
        raise ValueError(f"Image index {img_index} is out of range. Please provide a valid index between 0 and {total_img - 1}.")
    else:
        cam_position = cam_position[img_index] 
        cam_position = cam_position[:, :2] #2D  (x,y)
        T = cam_position.cpu().numpy() 
        T = torch.tensor(T, dtype=torch.float32).to(device)  

    return T

if __name__ == "__main__":
    cam_pose = cam_pose_finder(images)
    img_index = 0
    coords = pose_to_coords(cam_pose, img_index=img_index)
    print(f"Camera pose of img {img_index}, coord shape {coords.shape} is {coords}")

    torch.save(coords, f"cam_pose_coords_img{img_index}.pt")
    print(f"Camera pose of img {img_index} saved to vggt_cam_pose.pt")
