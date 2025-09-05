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
robot.speed_index = 2  # Start at fast
keyboard = KeyboardTeleop(keyboard_config)

# To connect you already should have this script running on LeKiwi: `python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=my_awesome_kiwi`
robot.connect()
keyboard.connect()

_init_rerun(session_name="lekiwi_teleop")

# initializations
remaining_x_time = 0.0
remaining_y_time = 0.0

initialise = True
frame_count = 0

# Get vggt distance
values = []
with open(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../sonata/dis_output.py")), "r"
) as f:
    for line in f:
        line = line.strip()
        if line and "=" in line:
            values.append(float(line.split("=")[1].strip()))

distance_x, distance_y = values
distance_y = -distance_y
print(f"distance x: {distance_x}, distance y: {distance_y}")

FPS = 30 

prev_time = time.perf_counter()
while True:
    # Measure full cycle interval
    current_time = time.perf_counter()
    interval = current_time - prev_time
    prev_time = current_time

    # Sequential timers
    if remaining_x_time > 0:
        remaining_x_time = max(0.0, remaining_x_time - interval)
    elif remaining_y_time > 0:
        remaining_y_time = max(0.0, remaining_y_time - interval)

    observation = robot.get_observation()

    frame_count += 1

    # Freeze arm pose
    arm_action = {
        "arm_shoulder_pan.pos": 23.299418604651152,
        "arm_shoulder_lift.pos": -5.887372,
        "arm_elbow_flex.pos": -76.09562,
        "arm_wrist_flex.pos": 30.989012,
        "arm_wrist_roll.pos": -90.42735,
        "arm_gripper.pos": 98.77133,
    }

    keyboard_keys = keyboard.get_action()

    if initialise:
        base_action,xy_speed, x_duration, y_duration= robot._from_keyboard_to_base_action_vggt(
            pressed_keys=keyboard_keys, 
            dis_y=distance_y, 
            dis_x=distance_x
            )

        remaining_x_time = 1.5 * x_duration
        remaining_y_time = 1.5 * y_duration

        initialise = False

    # Define outside the loop
    click = True
    click_before = 0
    
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
        click_before = current_time  

    if not click and (current_time - click_before) >= 3:
        click = True


    # Sequential motion: X -> Y
    if remaining_x_time > 0:
        # Move forward along X
        base_action = {"x.vel": xy_speed, "y.vel": 0.0, "theta.vel": 0.0}
    elif remaining_y_time > 0:
        # Move along Y
        base_action = {"x.vel": 0.0, "y.vel": xy_speed, "theta.vel": 0.0}
    else:
        base_action = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}

    log_rerun_data(observation, {**base_action})
    action = {**arm_action, **base_action}
    robot.send_action(action)
    print(
        f"Remaining X: {remaining_x_time:.2f}s, Y: {remaining_y_time:.2f}s\n"
    )

    busy_wait(1.0 / FPS)

    if remaining_x_time == 0.0 and remaining_y_time == 0.0:
        print("Motion complete. Robot stopped.")
        break
