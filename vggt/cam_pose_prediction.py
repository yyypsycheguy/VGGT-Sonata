from vggt.utils.pose_enc import extri_intri_to_pose_encoding
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.models.vggt import VGGT
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

model.eval()

# Process images
image_names = []
for img in os.listdir("images"):
    if img.endswith((".jpg", ".jpeg", ".png")):
        image_names.append(os.path.join("images", img))

images = load_and_preprocess_images(image_names).to(device)


def cam_pose_finder(images, data=torch.load("vggt_raw_output.pt")):

    '''Returns:
    torch.Tensor: Encoded camera pose parameters with shape BxSx9.
        For "absT_quaR_FoV" type, the 9 dimensions are:
        - [:3] = absolute translation vector T (3D)
        - [3:7] = rotation as quaternion quat (4D)
        - [7:] = field of view (2D)'''
    
    extrinsic = data["extrinsic"]  # [B, S, 3, 4]
    intrinsic = data["intrinsic"]  # [B, S, 3, 3]
    image_size_hw = images.shape[-2:]
    cam_pose = extri_intri_to_pose_encoding(
        extrinsic, 
        intrinsic, 
        image_size_hw, 
    )

    return cam_pose

if __name__ == "main":
    cam_pose = cam_pose_finder(images)
    print(f"Camera pose shape: {cam_pose.shape}")

    torch.save(cam_pose, "vggt_cam_pose.pt")
    print(f"Camera pose saved to vggt_cam_pose.pt")
