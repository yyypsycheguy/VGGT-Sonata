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

remaining_x_time = 0.0
remaining_y_time = 0.0
remaining_theta_time = 0.0

initialise = True

# Get vggt distance
values = []
with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../sonata/dis_output.py")), 'r') as f:
    for line in f:
        line = line.strip()
        if line and '=' in line:
            values.append(float(line.split('=')[1].strip()))

distance_x, distance_y = values
distance_y = -distance_y
print(f"distance x: {distance_x}, distance y: {distance_y}")

frame_count = 0
save_every_n_frames = 70  # Save every 5 frames
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
    elif remaining_theta_time > 0:
        remaining_theta_time = max(0.0, remaining_theta_time - interval)
    elif remaining_y_time > 0:
        remaining_y_time = max(0.0, remaining_y_time - interval)

    observation = robot.get_observation()

    if frame_count % save_every_n_frames == 0:
        wrist_image = observation["wrist"]
        # Save image for VGGT
        vggt_img_folder = "../../../vggt/images"
        os.makedirs(vggt_img_folder, exist_ok=True)
        vggt_image_path = os.path.join(vggt_img_folder, f"{time.strftime('%Y_%m_%d_%H:%M:%S')}.jpg")
        cv2.imwrite(vggt_image_path, wrist_image)
        print(f"Saved wrist camera image to {vggt_image_path}")

    frame_count += 1

    # Freeze arm pose
    arm_action = {
        "arm_shoulder_pan.pos": 23.299418604651152,
        "arm_shoulder_lift.pos": -5.021645021645028,
        "arm_elbow_flex.pos": -77.10131758291686,
        "arm_wrist_flex.pos": 0.2439024390243958,
        "arm_wrist_roll.pos": -30.793650793650798,
        "arm_gripper.pos": 98.67424242424242,
    }

    if initialise:
        keyboard_keys = keyboard.get_action()
        base_action, xy_speed, theta_speed, x_duration, y_duration, theta_duration = robot._from_keyboard_to_base_action_vggt(
            pressed_keys=keyboard_keys,
            dis_y=distance_y,
            dis_x=distance_x
        )

        remaining_x_time = 1.5 * x_duration
        remaining_theta_time = theta_duration
        remaining_y_time = 1.5 *y_duration

        initialise = False

    # Sequential motion: X -> theta -> Y
    if remaining_x_time > 0:
        # Move forward along X
        base_action = {"x.vel": xy_speed, "y.vel": 0.0, "theta.vel": 0.0}
    elif remaining_theta_time > 0:
        # Rotate
        target_angle = 90 if distance_y > 0 else -90 if distance_y < 0 else 0
        base_action = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": theta_speed if target_angle > 0 else -theta_speed}
    elif remaining_y_time > 0:
        # Move along new heading (X axis after rotation)
        base_action = {"x.vel": xy_speed, "y.vel": 0.0, "theta.vel": 0.0}
    else:
        base_action = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}

    log_rerun_data(observation, {**base_action})
    action = {**arm_action, **base_action}
    robot.send_action(action)
    print(f"Remaining X: {remaining_x_time:.2f}s, Theta: {remaining_theta_time:.2f}s, Y: {remaining_y_time:.2f}s\n")

    busy_wait(1.0 / FPS)

    if remaining_x_time == 0.0 and remaining_theta_time == 0.0 and remaining_y_time == 0.0:
        print("Motion complete. Robot stopped.")
        break
