# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

import concavity
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
from concavity.utils import *
from scipy.spatial import Delaunay
from scipy.stats import zscore
from shapely.geometry import MultiLineString, MultiPoint, Polygon
from shapely.ops import polygonize, unary_union
from sklearn.cluster import DBSCAN

import sonata

# on;y uncomment when using reachy
'''vggt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../vggt"))
sys.path.append(vggt_path)

from share_var import cam_frame_dis

print(f"Loaded: cam_frame_dis={cam_frame_dis}")'''

try:
    import flash_attn
except ImportError:
    flash_attn = None


# ScanNet Meta data
VALID_CLASS_IDS_20 = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    14,
    16,
    24,
    28,
    33,
    34,
    36,
    39,
)


CLASS_LABELS_20 = (
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refrigerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
    "otherfurniture",
)

SCANNET_COLOR_MAP_20 = {
    0: (0.0, 0.0, 100.0),
    1: (174.0, 199.0, 232.0),
    2: (152.0, 223.0, 138.0),
    3: (31.0, 119.0, 180.0),
    4: (255.0, 187.0, 120.0),
    5: (188.0, 189.0, 34.0),
    6: (140.0, 86.0, 75.0),
    7: (255.0, 152.0, 150.0),
    8: (214.0, 39.0, 40.0),
    9: (197.0, 176.0, 213.0),
    10: (148.0, 103.0, 189.0),
    11: (196.0, 156.0, 148.0),
    12: (23.0, 190.0, 207.0),
    14: (247.0, 182.0, 210.0),
    15: (66.0, 188.0, 102.0),
    16: (219.0, 219.0, 141.0),
    17: (140.0, 57.0, 197.0),
    18: (202.0, 185.0, 52.0),
    19: (51.0, 176.0, 203.0),
    20: (200.0, 54.0, 131.0),
    21: (92.0, 193.0, 61.0),
    22: (78.0, 71.0, 183.0),
    23: (172.0, 114.0, 82.0),
    24: (255.0, 127.0, 14.0),
    25: (91.0, 163.0, 138.0),
    26: (153.0, 98.0, 156.0),
    27: (140.0, 153.0, 101.0),
    28: (158.0, 218.0, 229.0),
    29: (100.0, 125.0, 154.0),
    30: (178.0, 127.0, 135.0),
    32: (146.0, 111.0, 194.0),
    33: (44.0, 160.0, 44.0),
    34: (112.0, 128.0, 144.0),
    35: (96.0, 207.0, 209.0),
    36: (227.0, 119.0, 194.0),
    37: (213.0, 92.0, 176.0),
    38: (94.0, 106.0, 211.0),
    39: (82.0, 84.0, 163.0),
    40: (100.0, 85.0, 144.0),
}

CLASS_COLOR_20 = [SCANNET_COLOR_MAP_20[id] for id in VALID_CLASS_IDS_20]


class SegHead(nn.Module):
    def __init__(self, backbone_out_channels, num_classes):
        super(SegHead, self).__init__()
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)

    def forward(self, x):
        return self.seg_head(x)


if __name__ == "__main__":
    # set random seed
    sonata.utils.set_seed(24525867)
    # Load model
    if flash_attn is not None:
        model = sonata.load("sonata", repo_id="facebook/sonata").cuda()
    else:
        custom_config = dict(
            enc_patch_size=[1024 for _ in range(5)],  # reduce patch size if necessary
            enable_flash=False,
        )
        model = sonata.load(
            "sonata", repo_id="facebook/sonata", custom_config=custom_config
        ).cuda()
    # Load linear probing seg head
    ckpt = sonata.load(
        "sonata_linear_prob_head_sc", repo_id="facebook/sonata", ckpt_only=True
    )
    seg_head = SegHead(**ckpt["config"]).cuda()
    seg_head.load_state_dict(ckpt["state_dict"])

    # Load data transform pipline
    config = [
        dict(type="CenterShift", apply_z=True),
        dict(
            type="GridSample",
            grid_size=0.02,
            hash_type="fnv",
            mode="train",
            return_grid_coord=True,
            return_inverse=True,
        ),
        dict(type="NormalizeColor"),
        dict(type="ToTensor"),
        dict(
            type="Collect",
            keys=("coord", "grid_coord", "color", "inverse"),
            feat_keys=("coord", "color", "normal"),
        ),
    ]
    transform = sonata.transform.Compose(config)
    # transform = sonata.transform.default()

    # Load data
    #point = torch.load("../vggt/predictions.pt")
    point = torch.load("../vggt/predictions.pt")
    print(point.keys())
    point["coord"] = point["coord"].numpy()
    print(f"Loaded point cloud with {len(point['coord'])} points")

    original_coord = point["coord"].copy()
    point = transform(point)

    # Inference
    model.eval()
    seg_head.eval()
    with torch.inference_mode():
        for key in point.keys():
            if isinstance(point[key], torch.Tensor):
                point[key] = point[key].cuda(non_blocking=True)
        # model forward:
        point = model(point)
        while "pooling_parent" in point.keys():
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent

        feat = point.feat
        seg_logits = seg_head(feat)
        pred = seg_logits.argmax(dim=-1).data.cpu().numpy()
        color = np.array(CLASS_COLOR_20)[pred]
        name = np.array(CLASS_LABELS_20)[pred]
        print(f"Predicted {len(np.unique(pred))} classes, {np.unique(pred)}")
        print(f"Predicted classes: {np.unique(name)}")

    # Visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point.coord.cpu().detach().numpy())
    pcd.colors = o3d.utility.Vector3dVector(color / 255)
    o3d.visualization.draw_geometries([pcd])
    # o3d.io.write_point_cloud("sem_seg.ply", pcd)

    # Save results
    torch.save(point, "results.pt")
    print("Results saved to results.pt")
    print(point.keys())


    # ------------ get coords of selected classes ------------
    def get_coords_by_class(point, class_name):
        """
        Get coordinates of points belonging to a specific class.
        Args:
            point (dict): Dictionary containing point cloud data.
            class_name (str): Name of the class to filter by.
        Returns:
            np.ndarray: Coordinates of points belonging to the specified class.
        """
        coords = point["coord"].cpu().detach().numpy()
        mask = np.array([name[i] == class_name for i in range(len(name))])
        return coords[mask]


    # get coords of target class
    frame_dis = 1.45 #3.1
    floor_coords = get_coords_by_class(point, "chair")
    print(f"\n Max chair coords: {max(floor_coords[:, 2])}, min chair coords: {min(floor_coords[:, 2])}")
    max_index = np.argmax(floor_coords[:, 2])
    max_coord = floor_coords[max_index]
    print(f"Original max chair coord: {max_coord}, index: {max_index}")
    print(f"Calibrated max chair depth: {max_coord[2] + frame_dis}, index: {max_index}")
    

    # get overall min coord floor
    floor_coords = get_coords_by_class(point, "floor")
    print(f"\n Max floor coords: {max(floor_coords[:, 2])}, min floor coords: {min(floor_coords[:, 2])}")
    min_index = np.argmin(floor_coords[:, 2])
    min_coord = floor_coords[min_index]
    print(f"Original min floor coord: {min_coord}, index: {min_index}")
    print(f"Calibrated min floor depth: {min_coord[2] + frame_dis}, index: {min_index}")


    def scale_coord(frame_dis: float, min_depth= min_coord[2]) -> float:
        sf = frame_dis / min_depth/ 1000
        return sf
    

    path = os.path.join(os.path.dirname(__file__), "../vggt/share_var.py")
    os.path.abspath(path)
    with open(os.path.abspath(path), 'r') as f:
            content = f.read().strip()

    scale_factor = float(content.split('=')[1].strip())
    print("Old Scale factor:", scale_factor)
    

    scale_factor = scale_coord(frame_dis=frame_dis)
    print(f'Scaled by frame dis {frame_dis} m, Scale factor: {scale_factor}\n')


    # Save updated scale factor to share_var.py
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../vggt/share_var.py"))
    with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "../vggt/share_var.py")), 'w') as f:
        f.write(f'scale_factor = {scale_factor}\n')