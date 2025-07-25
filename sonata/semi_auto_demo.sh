#!/bin/bash
cd ../vggt
source .venv/bin/activate
uv run video-split-test.py
MAX_JOB=4 uv run vggt_inference_floor.py
deactivate

cd../sonata
source sonata-venv/bin/activate
export PYTHONPATH=./ 
 MAX_JOB=4 uv run inference_visualize-sonata.py 
deactivate

python3 <<EOF
from reachy2_sdk import ReachySDK
reachy = ReachySDK(host="172.18.131.66")
reachy.mobile_base.turn_on()
reachy.mobile_base.reset_odometry()
reachy.mobile_base.goto(x=4.462387561798096, y=-1.2463709115982056, theta=0)
print('Move complete.')reachy.mobile_base.reset_odometry()
EOFecho 'Max chair coord before +3.1: [ 1.8598803 -2.143104   2.9115975]'
echo 'Max chair z: 6.011597633361816, y: -2.143104076385498'
echo 'Max wall coord before +3.1: [ 1.8598803 -2.143104   2.9115975]'
echo 'Max wall z: 6.011597633361816, y: -2.143104076385498'
