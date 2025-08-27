import time
import sys
import os
import json
import cv2

from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.so100_leader import SO100Leader, SO100LeaderConfig
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data

FPS = 30

# Create the robot and teleoperator configurations
robot_config = LeKiwiClientConfig(remote_ip="172.18.134.136", id="my_lekiwi")
teleop_arm_config = SO100LeaderConfig(port="/dev/ttyACM0", id="my_awesome_leader_arm")
keyboard_config = KeyboardTeleopConfig(id="my_laptop_keyboard")

robot = LeKiwiClient(robot_config)
leader_arm = SO100Leader(teleop_arm_config)
keyboard = KeyboardTeleop(keyboard_config)

# To connect you already should have this script running on LeKiwi: `python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=my_awesome_kiwi`
robot.connect()
leader_arm.connect()
keyboard.connect()

_init_rerun(session_name="lekiwi_teleop")

freeze_pose = True

remaining_x_time = 0.0
remaining_theta_time = 0.0

# Get vggt distance
values = []
with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../sonata/dis_output.py")), 'r') as f:
    for line in f:
        line = line.strip()
        if line and '=' in line:  # skip empty lines
            values.append(float(line.split('=')[1].strip()))

lekiwi_dis_x, lekiwi_dis_y = values
print(lekiwi_dis_y, lekiwi_dis_x)


prev_time = time.perf_counter()
while True:
    # Measure full cycle interval
    current_time = time.perf_counter()
    interval = current_time - prev_time
    prev_time = current_time 

    remaining_x_time = max(0.0, remaining_x_time - interval)
    remaining_theta_time = max(0.0, remaining_theta_time - interval)

    observation = robot.get_observation()

    # Freeze arm pose
    arm_action = leader_arm.get_action()
    arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
    if freeze_pose:
        arm_action = {"arm_shoulder_pan.pos": -14.752906976744185, "arm_shoulder_lift.pos": -8.91774891774891, "arm_elbow_flex.pos": -68.37800999545661, "arm_wrist_flex.pos": 67.8048780487805, "arm_wrist_roll.pos": 6.764346764346769, "arm_gripper.pos": 98.95833333333334}

    # Save wrist camera image
    wrist_image = observation["wrist"]
    wrist_folder = 'wrist_images'
    os.makedirs(wrist_folder, exist_ok=True)
    wrist_image_path = os.path.join(wrist_folder, f"{time.strftime('%Y_%m_%d_%H:%M:%S')}.jpg")
    cv2.imwrite(wrist_image_path, wrist_image)

    # Save image for VGGT
    vggt_img_folder = "../../../vggt/images"
    os.makedirs(vggt_img_folder, exist_ok=True)
    vggt_image_path = os.path.join(vggt_img_folder, f"{time.strftime('%Y_%m_%d_%H:%M:%S')}.jpg")
    cv2.imwrite(vggt_image_path, wrist_image)

    keyboard_keys = keyboard.get_action()

    base_action, xy_speed, theta_speed, remaining_x_time, remaining_theta_time = robot._from_keyboard_to_base_action_vggt(
        pressed_keys=keyboard_keys,
        dis_y=lekiwi_dis_y,
        dis_x=lekiwi_dis_x
    )

    # Keep moving until both times finish
    if remaining_x_time > 0 or remaining_theta_time > 0:
        base_action["x.vel"] = xy_speed if remaining_x_time > 0 else 0.0
        base_action["theta.vel"] = theta_speed if remaining_theta_time > 0 else 0.0
    else:
        base_action = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}

    action = {**arm_action, **base_action}
    robot.send_action(action)
    print(f"Remaining X: {remaining_x_time:.2f}s, Remaining Theta: {remaining_theta_time:.2f}s")

    busy_wait(1.0 / FPS)

    if remaining_x_time == 0.0 and remaining_theta_time == 0.0:
        print("Motion complete. Robot stopped.")
        break