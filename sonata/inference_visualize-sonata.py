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


import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import sonata

from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint
from scipy.stats import zscore

import concavity
from concavity.utils import *
from scipy.stats import zscore

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
    #transform = sonata.transform.default()

    # Load data
    point = torch.load("../vggt/predictions.pt")
    print(point.keys())
    point["coord"] = point["coord"].numpy()  # Ensure coordinates are float
    print(f"Loaded point cloud with {len(point['coord'])} points")
    

    # point.pop("segment200")
    # segment = point.pop("segment20")
    # point["segment"] = torch.zeros_like()  # two kinds of segment exist in ScanNet, only use one
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
    #print(point)

    # ------------ get the floor points ------------
    floor_points = []
    coords_array = point["coord"].cpu().detach().numpy()
    for i, coords in enumerate(coords_array):
        if name[i] == "floor":
            floor_points.append(coords)
    floor_points = np.array(floor_points)
    print(f"Found {len(floor_points)} floor points")
    print(f"Floor points: {np.unique(floor_points, axis=0)}")

    #show visualisation
    floor_pcd = o3d.geometry.PointCloud()
    floor_pcd.points = o3d.utility.Vector3dVector(floor_points)
    floor_pcd.colors = o3d.utility.Vector3dVector(
        np.array([0.0, 1.0, 0.0]).reshape(1, 3)
    )  # green color for floor
    o3d.visualization.draw_geometries([floor_pcd])


    # ------------ Compute convex hull (borders) of floor points ------------

    def filter_largest_cluster(points_2d: np.ndarray, eps=0.2, min_samples=30) -> np.ndarray:
        """
        Removes all but the largest cluster of 2D points using DBSCAN.

        Args:
            points_2d (np.ndarray): [N, 2] array of 2D (x, z) points.
            eps (float): DBSCAN neighborhood radius.
            min_samples (int): Minimum number of points per cluster.

        Returns:
            np.ndarray: Filtered points belonging to the largest cluster.
        """
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points_2d)
        labels = clustering.labels_

        # -1 is noise in DBSCAN
        unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)

        if len(counts) == 0:
            raise ValueError("No valid clusters found by DBSCAN.")

        largest_cluster_label = unique_labels[np.argmax(counts)]
        largest_cluster_mask = labels == largest_cluster_label
        return points_2d[largest_cluster_mask]

    def alpha_shape(points, alpha=0.0001):
        """
        Compute the alpha shape (concave hull) of a set of 2D points.
        Args:
            points (np.ndarray): Nx2 array of (x, z) coordinates.
            alpha (float): Alpha value to control the detail. Smaller = tighter shape.
        Returns:
            shapely.geometry.Polygon: The resulting concave polygon.
        """
        from shapely.geometry import Polygon, MultiLineString
        from shapely.ops import unary_union, polygonize
        from scipy.spatial import Delaunay

        if len(points) < 4:
            return MultiPoint(points).convex_hull

        tri = Delaunay(points)
        triangles = points[tri.simplices]
        
        a_shape_edges = []
        for tri_coords in triangles:
            a, b, c = tri_coords
            len_ab = np.linalg.norm(a - b)
            len_bc = np.linalg.norm(b - c)
            len_ca = np.linalg.norm(c - a)
            s = (len_ab + len_bc + len_ca) / 2.0
            area = np.sqrt(s * (s - len_ab) * (s - len_bc) * (s - len_ca))
            circum_r = len_ab * len_bc * len_ca / (4.0 * area + 1e-8)

            if circum_r < 1.0 / alpha:
                a_shape_edges += [(tuple(a), tuple(b)), (tuple(b), tuple(c)), (tuple(c), tuple(a))]

        m = MultiLineString(a_shape_edges)
        polygons = list(polygonize(unary_union(m)))
        return unary_union(polygons)

    def extract_floor_trajectory(floor_points_3d: np.ndarray, zscore_threshold=1.5, show_plot=True, save_path="floor_trajectory.png"):
        """
        Extracts the border trajectory of the floor from 3D points with outlier removal.

        Args:
            floor_points_3d (np.ndarray): Nx3 array of 3D (x, y, z) (right, forward, up) floor points.
            zscore_threshold (float): Z-score threshold for filtering outliers.
            show_plot (bool): Whether to plot the result.
            save_path (str or None): If set, saves the figure to this path.

        Returns:
            trajectory_2d (np.ndarray): Mx2 array of 2D (x, z) border points.
        """
        # Project to 2D (X-Y plane)
        floor_points_2d = floor_points_3d[:, :2]  # shape: [N, 2] [X,Y]

        # Remove outliers using z-score
        zs = zscore(floor_points_2d, axis=0)
        mask = np.all(np.abs(zs) < zscore_threshold, axis=1)
        filtered_points = floor_points_2d[mask]

        filtered_points = filter_largest_cluster(filtered_points, eps=0.1, min_samples=10)
        print(f"Filtered points shape: {filtered_points.shape}")

        if len(filtered_points) < 3:
            raise ValueError("Not enough inlier points after filtering for convex hull.")
        
        trajectory_2d = concavity.concave_hull(filtered_points, 50)
        trajectory_2d = trajectory_2d.buffer(-0.27)
        trajectory_2d = np.array(trajectory_2d.exterior.coords)


        if show_plot:
            plt.figure(figsize=(8, 6))
            plt.plot(floor_points_2d[:, 0], floor_points_2d[:, 1], 'go', alpha=0.2, label='Raw Points')
            plt.plot(filtered_points[:, 0], filtered_points[:, 1], 'bo', alpha=0.4, label='Filtered Points')
            plt.plot(np.append(trajectory_2d[:, 0], trajectory_2d[0, 0]),
                    np.append(trajectory_2d[:, 1], trajectory_2d[0, 1]),
                    'r-', lw=2, label='Trajectory')
            plt.fill(trajectory_2d[:, 0], trajectory_2d[:, 1], 'r', alpha=0.1)
            plt.title("Floor Border Trajectory (After Outlier Removal)")
            plt.xlabel("Y (forward)")
            plt.ylabel("X(right)")
            #plt.axis('equal')
            plt.grid(True)
            plt.legend()
            if save_path:
                plt.savefig(save_path)
                print(f"Plot saved to {save_path}")
            plt.close()

        return trajectory_2d


    trajectory = extract_floor_trajectory(floor_points)
<<<<<<< Updated upstream
<<<<<<< Updated upstream
    print("Trajectory points (X-Y):")
=======
    print("Trajectory points (X-Y)")
>>>>>>> Stashed changes
=======
    print("Trajectory points (X-Y)")
>>>>>>> Stashed changes
