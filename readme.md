# Using CV models: VGGT-Sonata pipeline for zero-shot robotics spatial navigation

This repository proposes a pipeline that enables 3D scene reconstruction from one single image or a video, with a simple RBG camera predict object depth, and achieve semantic segmentation of objects in the scene. It does this by running VGGT model inference, and feeds the output into Sonata which does the segmentation. Then in the repository, we use Lekiwi robot from LeRobot Hugging Face to autonomously have lekiwi navigate to any given target object.

Before running the pipeline, remember to set up dependencies for VGGT and Sonata, as well as lekiwi.

## Setup for [VGGT] (https://github.com/facebookresearch/vggt.git) -- Visual Geometry Grounded Transformer
```
cd vggt
uv pip install -r requirements.txt
uv pip install -r requirements_demo.txt
uv venv .venv

# Install a package in the new virtual environment if needed
uv pip install ruff

# to activate 
source .venv/bin/activate
```

## Setup for [Sonata] (https://github.com/facebookresearch/sonata.git) 
```
uv venv sonata-venv
source sonata-venv/bin/activate

# Ensure Cuda and Pytorch are already installed in your local environment

# CUDA_VERSION: cuda version of local environment (e.g., 124), check by running 'nvcc --version'
# TORCH_VERSION: torch version of local environment (e.g., 2.5.0), check by running 'python -c "import torch; print(torch.__version__)"'
pip install spconv-cu${CUDA_VERSION}
pip install torch-scatter -f https://data.pyg.org/whl/torch-{TORCH_VERSION}+cu${CUDA_VERSION}.html
pip install git+https://github.com/Dao-AILab/flash-attention.git
pip install huggingface_hub timm

# (optional, or directly copy the sonata folder to your project)
python setup.py install
```

Please refer to original [Sonata] (https://github.com/facebookresearch/sonata.git) repository for trouble shooting.

## Setup for [lerobot directory](https://huggingface.co/docs/lerobot/installation) & run to do spatial navigation by targeting specific objects

If first time setting up lekiwi please refer to Hugging Face Lerobot installation guide first. To teleoperate and run spatial navigation of this repository, SSH into your Raspberry Pi, run this command:
```
cd lerobot
conda activate lerobot

# start teleoperating
python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=my_awesome_kiwi
```

Make sure you have set the correct remote_ip and port in examples/lekiwi/teleoperate.py. Then cd into folder containing Sonata, this is where the automation script is located:
```
# add script into your executable
chmod +x bash-command.sh

# run
./bash-command.sh 
```
This takes photo inputs from your lekiwi robot, runs inference with vggt & sonata, gives the input calibrated distance for lekiwi to execute.

# Parameter modification
If you are using other machines/robots to take input pictures, you have to modify this parameter: located at 
```
cd sonata/inference_visualize-sonata.py
```
which is the Sonata inference file. You need to manually measure frame_dis:
```
frame_dis = 1.45 # measured frame distance of le kiwi; around 3.1m for iPhone video recording, it is dis of blind angle from camera to bottom of video frame

# modify this as well
target = "chair"
```
into the furniture you would like to track, that is in the view of the camera and in the proposed categories of Sonata.