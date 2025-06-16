from vggt.utils.pose_enc import extri_intri_to_pose_encoding
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.models.vggt import VGGT
from vggt_inference_floor import extrinsic, intrinsic
import torch
import os


device = "cuda" if torch.cuda.is_available() else "cpu"

# Process images
image_names = []
for img in os.listdir("images"):
    if img.endswith((".jpg", ".jpeg", ".png")):
        image_names.append(os.path.join("images", img))

images = load_and_preprocess_images(image_names).to(device)


def cam_pose_finder(images, extrinsic, intrinsic):
    '''Returns:
    torch.Tensor: Encoded camera pose parameters with shape BxSx9.
    With "absT_quaR_FoV" type, the 9 dimensions are:
    - [:3] = absolute translation vector T (3D) camera's position in world coordinates
    - [3:7] = rotation as quaternion quat (4D) representing the camera's orientation
    - [7:] = field of view (2D) representing the camera's horizontal and vertical FoV angles '''

    # extrinsic [B, S, 3, 4]
    # intrinsic [B, S, 3, 3]
    image_size_hw = images.shape[-2:]
    cam_pose = extri_intri_to_pose_encoding(
        extrinsic, 
        intrinsic, 
        image_size_hw, 
    )
    cam_pose = cam_pose.squeeze(0)  # Remove batch dimension
    cam_pose = cam_pose.to(device) 

    return cam_pose


def pose_to_coords(pose, img_index):
    '''
    args: pose  = position encoding, output of cam_pose_finder
          img_index = image index in the series
    Returns: coordinates realtive to the origin (first image in the series).
    Convert camera pose to 3D coordinates.'''
    total_img = len(image_names)

    cam_position = pose[:,:3]  # Bx3 (x,y,z)
    #print(f"Camera 2D (x,y)(right, backward) position shape: {cam_position.shape}, values: {cam_position}")
    if img_index > total_img-1 or img_index < 0:
        raise ValueError(f"Image index {img_index} is out of range. Please provide a valid index between 0 and {total_img}- 1.")
    else:
        cam_position = cam_position[img_index,[0,2]] #2D  (x,y)
        T = cam_position.cpu().detach().numpy() 
        T = torch.tensor(T, dtype=torch.float32).to(device)  

    return T  

def relative_coords(pose, img_index):
    total_img = len(image_names)

    cam_position = pose[:,:3]  # Bx3 (x,y,z)
    #print(f"Camera 2D (x,y)(right, backward) position shape: {cam_position.shape}, values: {cam_position}")
    if img_index > total_img-1 or img_index < 0:
        raise ValueError(f"Image index {img_index} is out of range. Please provide a valid index between 0 and {total_img}- 1.")
    else:
        origin = cam_position[0,[0,2]]
        target = cam_position[img_index,[0,1]] #2D  (x,y)
        relative = target - origin 

        pixel_distance = torch.norm(relative) # 2D distance in pixels
        scale_factor = 7.5 / pixel_distance  # meters per unit
        scale_factor = 14.47
        print(f"pixel distance: {pixel_distance}, scale factor: {scale_factor}")

        x, y = target[0], target[1]
        x_m = x * scale_factor  
        y_m = y * scale_factor
        coords_m = torch.tensor([x_m, y_m], dtype=torch.float32).to(device)
    return coords_m # returns coords in meters, relative to origin



def coords_meter(coords):
    '''Convert coordinates from pixels to meters.
    args: coords = 2D coordinates in pixels (x,y)'''
    factor = 15 # m/coord unit
    x, y = coords[0], coords[1]
    x_m = x * factor 
    y_m = y * factor
    coords_m = torch.tensor([x_m, y_m], dtype=torch.float32).to(device)
    return coords_m



if __name__ == "__main__":
    pose = cam_pose_finder(images, extrinsic, intrinsic)
    print(f"Pose: {pose}")
    print(f"untranformed extrinsic: {extrinsic}")
    meter = relative_coords(pose, 6) 
    print(f"6 Coordinates in meters relative to the first image: {meter}")
    meter = relative_coords(pose, 5)
    print(f"5 Coordinates in meters relative to the first image: {meter}")