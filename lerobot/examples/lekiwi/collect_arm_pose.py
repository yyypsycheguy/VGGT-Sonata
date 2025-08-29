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

FPS = 1

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


while True:
    t0 = time.perf_counter()

    observation = robot.get_observation()

    arm_action = leader_arm.get_action()
    arm_action = {f"arm_{k}": v for k, v in arm_action.items()}

    # Save wrist camera image: uncomment for saving wrist camera images
    wrist_image = observation["wrist"]
    folder = 'wrist_images'
    os.makedirs(folder, exist_ok=True)
    wrist_image_path = os.path.join(folder, f"{time.strftime('%Y_%m_%d_%H:%M:%S')}.jpg")
    cv2.imwrite(wrist_image_path, wrist_image)
    print(f"Saved wrist camera image to {wrist_image_path}")

    # save action
    folder = 'actions'
    os.makedirs(folder, exist_ok=True)
    action_path = os.path.join(folder, "actions.txt")
    action_log = {
        "arm_action": arm_action
    }

    with open(action_path, 'a') as f:
        f.write(json.dumps(action_log) + "\n")
    print("Arm action appended:", arm_action)

    keyboard_keys = keyboard.get_action()
    base_action = robot._from_keyboard_to_base_action(keyboard_keys)

    log_rerun_data(observation, {**arm_action, **base_action})
    action = {**arm_action, **base_action} if len(base_action) > 0 else arm_action
    robot.send_action(action)

    interval = time.perf_counter() - t0
    busy_wait(max(1.0 / FPS - interval, 0.0))