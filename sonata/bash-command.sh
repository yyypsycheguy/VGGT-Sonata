#!/bin/bash

counter=0

#run vggt
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

counter=$((counter + 1))

# run another loop
cd ../vggt
source .venv/bin/activate
#uv run video-split-test.py
MAX_JOB=4 uv run vggt_inference_floor.py
deactivate

cd ../sonata
source sonata-venv/bin/activate
export PYTHONPATH=./
MAX_JOB=4 uv run inference_visualize-sonata.py

counter=$((counter + 1))
if [ $counter -eq 2 ]; then
    echo "Completed two iterations of the loop."
    echo "scale_factor=1.0" > ../vggt/share_var.py
    echo "scale_factor reset to 1.0 "
    exit 0
fi

# operate lekiwi
cd ../lerobot/examples/lekiwi
conda activate lerobot
MAX_JOB=4 uv run teleoperate.py
conda deactivate