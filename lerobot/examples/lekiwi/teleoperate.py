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
vggt_mode = True 

# test: modify to test vggt inputs manually
remaining_x_time = 0.0
remaining_theta_time = 0.0
with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "dis_output.py")), 'r') as f: 
    lekiwi_dis_y = float(f.readline().split('=')[1].strip()) # lekiwi takes y forward so switch y and x
    lekiwi_dis_x = float(f.readline().split('=')[1].strip())

while True:
    start_time = time.time()
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

    # write last image to vggt
    if time.time() - start_time <= 5:
        vggt_img_folder = "vggt/images"
        os.makedirs(vggt_img_folder, exist_ok=True)
        vggt_image_path = os.path.join(vggt_img_folder, f"{time.strftime('%Y_%m_%d_%H:%M:%S')}.jpg")
        cv2.imwrite(vggt_image_path, wrist_image)
        print(f"Saved vggt image to {vggt_image_path}")


    # save action: only enable this when the arm pose needs recalibration to film at a better angle 
    # folder = 'actions'
    # os.makedirs(folder, exist_ok=True)
    #action_path = os.path.join(folder, "actions.txt")
    # action_log = {
    #     "arm_action": arm_action
    # }

    # with open(action_path, 'a') as f:
    #     f.write(json.dumps(action_log) + "\n")
    # print("Arm action appended:", arm_action)

    if freeze_pose: # replace arm pose if needed
        arm_action = {'arm_shoulder_pan.pos': 13.299418604651152, 'arm_shoulder_lift.pos': -5.021645021645028, 'arm_elbow_flex.pos': -77.10131758291686, 'arm_wrist_flex.pos': 0.2439024390243958, 'arm_wrist_roll.pos': -30.793650793650798, 'arm_gripper.pos': 98.67424242424242}
    else:
        pass

    if vggt_mode:
        keyboard_keys = keyboard.get_action()

        # Only get a new movement command when both timers are 0
        if remaining_x_time <= 0 and remaining_theta_time <= 0:
            base_action, xy_speed, theta_speed, x_duration,theta_duration = robot._from_keyboard_to_base_action_vggt(
                pressed_keys=keyboard_keys,
                dis_y=lekiwi_dis_y,
                dis_x=lekiwi_dis_x,
            )
            remaining_x_time = x_duration
            remaining_theta_time = theta_duration
            print(f"New base action: {base_action}, x_duration: {x_duration:.2f}, theta_duration: {theta_duration:.2f}, xy_speed: {xy_speed}, theta_speed: {theta_speed}")
        else:
            # Continue previous motion
            pass


    if not vggt_mode or (remaining_x_time <= 0 and remaining_theta_time <= 0):
        keyboard_keys = keyboard.get_action()
        base_action = robot._from_keyboard_to_base_action(keyboard_keys)

    log_rerun_data(observation, {**arm_action, **base_action})
    action = {**arm_action, **base_action} if len(base_action) > 0 else arm_action
    robot.send_action(action)

    # Countdown timers based on loop duration
    interval = time.perf_counter() - t0
    if vggt_mode:
        if remaining_x_time > 0:
            remaining_x_time -= interval
            if remaining_x_time <= 0:
                base_action["x.vel"] = 0.0

        if remaining_theta_time > 0:
            remaining_theta_time -= interval
            if remaining_theta_time <= 0:
                base_action["theta.vel"] = 0.0

        print(f"b4 loop x speed: {xy_speed:.2f} m/s")
        remaining_theta_time = max(0.0, remaining_theta_time - interval)
        remaining_x_time = max(0.0, remaining_x_time - interval)

        if remaining_theta_time > 0: # rotation
            base_action["x.vel"] = 0.0
        elif remaining_x_time > 0: # forward
            base_action["theta.vel"] = 0.0
            base_action["x.vel"] = xy_speed
            print(f'x speed: {xy_speed:.2f} m/s')
        else:
            base_action["x.vel"] = 0.0
            base_action["theta.vel"] = 0.0

        print(f"Remaining theta: {remaining_theta_time:.2f}s, Remaining x: {remaining_x_time:.2f}s")

    busy_wait(max(1.0 / FPS - interval, 0.0))