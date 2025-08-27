#!/bin/bash
set -e  # exit on any error

# run vggt
cd ../vggt
source .venv/bin/activate
MAX_JOB=4 uv run vggt_inference.py
deactivate

# run sonata 
cd ../sonata
source sonata-venv/bin/activate
export PYTHONPATH=./
MAX_JOB=4 uv run sonata_inference.py
uv run scaling.py
deactivate

counter=$((counter + 1))

# run another loop
cd ../vggt
source .venv/bin/activate
CUDA_LAUNCH_BLOCKING=1 MAX_JOB=3 uv run vggt_inference.py
deactivate

cd ../sonata
source sonata-venv/bin/activate
export PYTHONPATH=./
CUDA_LAUNCH_BLOCKING=1 MAX_JOB=3 uv run sonata_inference.py
uv run get_target_distance.py
deactivate


echo "scale_factor=1.0" > ../vggt/share_var.py
echo "scale_factor reset to 1.0 "