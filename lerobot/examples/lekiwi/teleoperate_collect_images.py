import json
import os
import sys
import time

import cv2

from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.so100_leader import SO100Leader, SO100LeaderConfig
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data

# Create the robot and teleoperator configurations
robot_config = LeKiwiClientConfig(remote_ip="172.18.134.136", id="my_lekiwi")
keyboard_config = KeyboardTeleopConfig(id="my_laptop_keyboard")

robot = LeKiwiClient(robot_config)
keyboard = KeyboardTeleop(keyboard_config)

# To connect you already should have this script running on LeKiwi: `python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=my_awesome_kiwi`
robot.connect()
keyboard.connect()

_init_rerun(session_name="lekiwi_teleop")

freeze_pose = True
start_time = time.time()

frame_count = 0
save_every_n_frames = 90  # Save every 70 frames
FPS = 30

robot.speed_index = 0  # Start at fast

while True:
    t0 = time.perf_counter()

    observation = robot.get_observation()

    # Arm and base actions
    arm_action = {
        "arm_shoulder_pan.pos": 23.299418604651152,
        "arm_shoulder_lift.pos": -5.887372,
        "arm_elbow_flex.pos": -76.09562,
        "arm_wrist_flex.pos": 30.989012,
        "arm_wrist_roll.pos": -90.42735,
        "arm_gripper.pos": 98.77133,
    }

    # Define outside the loop
    click = True
    click_before = 0

    # Inside your loop
    keyboard_keys = keyboard.get_action()
    current_time = time.perf_counter()

    if "p" in keyboard_keys and click:
        print(f"Keyboard keys: {keyboard_keys}")
        wrist_image = observation["wrist"]
        wrist_image = cv2.cvtColor(wrist_image, cv2.COLOR_RGB2BGR)
        folder = "../vggt/images"
        os.makedirs(folder, exist_ok=True)
        vggt_image_path = os.path.join(folder, f"{time.strftime('%Y_%m_%d_%H:%M:%S')}_{frame_count}.jpg")
        cv2.imwrite(vggt_image_path, wrist_image)
        print(f"Saved wrist camera image to {vggt_image_path}")

        click = False
        click_before = current_time  # start cooldown

    # reset click after 0.5 s
    if not click and (current_time - click_before) >= 3:
        click = True

    base_action = robot._from_keyboard_to_base_action(keyboard_keys)

    log_rerun_data(observation, {**base_action})
    action = {**arm_action, **base_action} if len(base_action) > 0 else arm_action
    robot.send_action(action)

    # Maintain 30 FPS
    interval = time.perf_counter() - t0
    busy_wait(max(1.0 / FPS - interval, 0.0))
