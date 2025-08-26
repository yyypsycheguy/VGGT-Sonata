#!/bin/bash
set -e  # exit on any error

# lekiwi get images
# cd ../lerobot
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate lerobot
# python examples/lekiwi/teleoperate_collect_imgs.py
# conda deactivate

counter=0

# run vggt
cd ../vggt
source .venv/bin/activate
#uv run video-split-test.py
MAX_JOB=4 uv run vggt_inference_floor.py
deactivate

# run sonata 
cd ../sonata
source sonata-venv/bin/activate
export PYTHONPATH=./
MAX_JOB=4 uv run inference_visualize-sonata.py
deactivate

counter=$((counter + 1))

# run another loop
cd ../vggt
source .venv/bin/activate
#uv run video-split-test.py
CUDA_LAUNCH_BLOCKING=1 MAX_JOB=3 uv run vggt_inference_floor.py
deactivate

cd ../sonata
source sonata-venv/bin/activate
export PYTHONPATH=./
CUDA_LAUNCH_BLOCKING=1 MAX_JOB=3 uv run inference_visualize-sonata.py
deactivate

counter=$((counter + 1))
if [ $counter -eq 2 ]; then
    echo "Completed two iterations of the loop."
    echo "scale_factor=1.0" > ../vggt/share_var.py
    echo "scale_factor reset to 1.0 "
fi

# operate lekiwi
# cd ../lerobot
# conda activate lerobot
# python examples/lekiwi/teleoperate.py
# conda deactivate
