#!/bin/bash

#run vggt
cd ../vggt
source .venv/bin/activate
uv run video-split-test.py
#MAX_JOB=4 uv run vggt_inference_detail.py
MAX_JOB=4 uv run vggt_inference_floor.py
deactivate

# run sonata 
cd ../sonata
source sonata-venv/bin/activate
export PYTHONPATH=./
MAX_JOB=4 uv run inference_visualize-sonata.py