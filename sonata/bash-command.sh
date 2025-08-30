#!/bin/bash
set -e  # exit on any error

# 1.run vggt
cd ../vggt
source .venv/bin/activate
MAX_JOB=4 uv run vggt_inference.py
deactivate

# 2.run sonata 
cd ../sonata
source sonata-venv/bin/activate
export PYTHONPATH=./
MAX_JOB=4 uv run sonata_inference.py
# 3. compute scale factor
uv run compute_scale_factor.py

cd ../vggt
source .venv/bin/activate
# scale extrinsic
uv run scale_extrinsic.py
deactivate


# 4. scale point cloud
cd ../sonata
source sonata-venv/bin/activate
uv run scale_pointcloud.py
#5. get target distance
uv run get_target_distance.py
# 7. construct 2D map
uv run construct_2Dmap.py