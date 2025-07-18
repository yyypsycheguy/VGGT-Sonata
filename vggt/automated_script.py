import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pillow_heif
import torch
from jupyterthemes import jtplot


def extract_match_key(filename):
    match = re.match(r"(\d{2}:\d{2}:\d{2})_.*_(\d+)\.avif", filename)  # regrex for name
    if match:
        return f"{match.group(1)}_{match.group(2)}"
    return None


def depth_mean(cam_path="lux_cam", vggt_path="lux_vggt"):
    gather = []
    mean_loss = []
    mean_sf = []
    pillow_heif.register_avif_opener()

    cam_files = {
        extract_match_key(f): f for f in os.listdir(cam_path) if extract_match_key(f)
    }
    vggt_files = {
        extract_match_key(f): f for f in os.listdir(vggt_path) if extract_match_key(f)
    }
    matched_keys = set(cam_files.keys()) & set(vggt_files.keys())

    if not matched_keys:
        print("No matched files found.")
        return []

    for key in sorted(matched_keys):
        cam_file = cam_files[key]
        vggt_file = vggt_files[key]

        cam_file_path = os.path.join(cam_path, cam_file)
        vggt_file_path = os.path.join(vggt_path, vggt_file)

        print(f"Processing matched pair for key '{key}':")

        if not (
            pillow_heif.is_supported(cam_file_path)
            and pillow_heif.is_supported(vggt_file_path)
        ):
            print(f"  Skipping unsupported file pair: {key}")
            continue

        cam_avif = pillow_heif.open_heif(cam_file_path, convert_hdr_to_8bit=False)
        vggt_avif = pillow_heif.open_heif(vggt_file_path, convert_hdr_to_8bit=False)

        img_cam_array = np.asarray(cam_avif)
        img_vggt_array = np.asarray(vggt_avif)
        img_vggt_array = np.resize(img_vggt_array, img_cam_array.shape)

        img_vggt_array_shift = img_vggt_array >> 4
        img_cam_array_shift = img_cam_array >> 4

        # filter points to graph 
        joint_mask_graph = (
            (img_cam_array_shift <= 410) & (img_cam_array_shift > 0) &
            (img_vggt_array_shift <= 990) & (img_vggt_array_shift > 0)
        )
        joint_mask_graph = joint_mask_graph.reshape(480, 640)
        img_vggt_array_graph = img_vggt_array_shift[joint_mask_graph]
        img_cam_array_graph = img_cam_array_shift[joint_mask_graph]

        # -- LOSS FUNCTION
        def loss(cam_depth, vggt_depth, scale_factor, upper_bound_cam= 410,  upper_bound_vggt= 990, lower_bound=0):
            # Apply filter
            joint_mask = (
                (cam_depth <= upper_bound_cam) & (cam_depth > 0) &
                (scale_factor * vggt_depth <= upper_bound_vggt) & (scale_factor * vggt_depth > lower_bound)
            )

            #joint_mask = joint_mask.reshape(480, 640) 
            masked = ((scale_factor * vggt_depth) - cam_depth)[joint_mask] 
            rmse = np.sqrt(np.mean(masked ** 2)) / 1000

            print(f' vggt after scaling: {np.min(scale_factor * vggt_depth):.4f} / {np.max(scale_factor * vggt_depth):.4f}')
            print(f"Scale: {scale_factor:.2f}, Masked points: {np.sum(joint_mask)}")
            print(f"vggt depth min/max (scaled): {np.min(scale_factor * vggt_depth):.4f} / {np.max(scale_factor * vggt_depth):.4f}")
            print(f"cam depth min/max: {np.min(cam_depth):.4f} / {np.max(cam_depth):.4f}")

            return rmse

        # --- aggregate loss
        loss_list = []
        for factor in np.arange(0, 1, 0.01):
            loss_value = loss(
                img_cam_array_shift, img_vggt_array_shift, scale_factor=factor
            )
            loss_value = np.mean(loss_value) # loss for each image
            loss_list.append(loss_value) # append loss value for each scale factor

        loss_list = np.array(loss_list)
        loss_list[np.isnan(loss_list)] = np.inf

        # Find vertex of loss curve
        def find_vertex(loss_list):
            idx = np.argmin(loss_list)
            return loss_list[idx], idx / 100

        vertex, sf = find_vertex(loss_list) # vertex is optimal loss

    print(f"\nFinal Mean Loss list: {loss_list}")
    aggregated_mean = np.mean(mean_loss)

    print(f"Aggregated Mean Loss: {vertex:.7f}")
    aggregated_mean = np.mean(mean_loss)
    print(f"Scale Factor: {np.mean(sf):.3f}")

    print(f"vggt shape: {img_vggt_array_shift.shape}")
    print(f"cam shape: {img_cam_array_shift.shape}")
    return (
        aggregated_mean,
        img_vggt_array_shift,
        img_cam_array_shift,
        img_vggt_array_graph,
        img_cam_array_graph,
        joint_mask_graph,
        mean_sf,
    )


if __name__ == "__main__":
    
    # mean_loss = depth_mean()
    (
        aggregated_mean,
        img_vggt_array_shift,
        img_cam_array_shift,
        img_vggt_array_graph,
        img_cam_array_graph,
        joint_mask_graph,
        mean_sf,
    ) = depth_mean()

    # onedork | grade3 | oceans16 | chesterish | monokai | solarizedl | solarizedd
    jtplot.style(theme='onedork', context='notebook', ticks=True, grid=True, figsize=(10, 6))

    masked_cam = np.full((480, 640), np.nan)
    masked_vggt = np.full((480, 640), np.nan)

    masked_cam[joint_mask_graph] = img_cam_array_shift[joint_mask_graph]
    masked_vggt[joint_mask_graph] = img_vggt_array_shift[joint_mask_graph]

    diff = (masked_vggt - masked_cam) / 1000
    plt.imshow(diff, cmap='RdYlGn_r', vmin=0, vmax=1)
    plt.colorbar()
    plt.title('Difference between VGGt and Cam (masked)')
    plt.savefig('vggt_cam_diff_masked.png', dpi=300)
    plt.show()
    print(f"figure saved as vggt_cam_diff_masked.png")

    plt.imshow((img_vggt_array_shift - img_cam_array_shift)/1000,  cmap='RdYlGn_r', vmin=0, vmax= 1)
    plt.colorbar()
    plt.title('Difference between VGGt and GT Depth Maps: before filtering')
    plt.savefig('vggt_cam_diff_before_filtering.png', dpi=300)
    plt.show()
    print(f"figure saved as vggt_cam_diff_before_filtering.png")


