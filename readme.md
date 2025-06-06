# FOR VGGT:
## run inference
```
source .venv/bin/activate
uv run vggt_inference.py
```

## run Viser 3D viewer:
```
python demo_viser.py --image_folder path/to/your/images/folder
uv run python demo_viser.py --image_folder images
```
---------------------------------------------------------------------------------------------------

# For Sonata:
## Run Sonata segmentation sample
```
cd sonata
export PYTHONPATH=./
uv run inference_visualize-sonata.py 
```

input strucutre expected by sonata:
point = {
  "coord": numpy.array,  # (N, 3)
  "color": numpy.array,  # (N, 3)
  "normal": numpy.array,  # (N, 3)
  "batch": numpy.array,  # (N,) optional
  "segment": numpy.array,  # (N,) optional
}
call sonata.data.load ("sample") to transform 
