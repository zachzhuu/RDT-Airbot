#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""

import argparse
import sys
import threading
import time
import yaml
from collections import deque

import numpy as np
import torch
from PIL import Image as PImage
import cv2

from scripts.airbot_model import create_model

import pyrealsense2 as rs
import airbot
import queue

# sys.path.append("./")

CAMERA_NAMES = ['cam_high', 'cam_right_wrist']

observation_window = None

lang_embeddings = None

# debug
preload_images = None


# Initialize the model
def make_policy(args):
    with open(args.config_path, "r") as fp:
        config = yaml.safe_load(fp)
    args.config = config

    # pretrained_text_encoder_name_or_path = "google/t5-v1_1-xxl"
    pretrained_vision_encoder_name_or_path = "google/siglip-so400m-patch14-384"
    model = create_model(
        args=args.config,
        dtype=torch.bfloat16,
        pretrained=args.pretrained_model_name_or_path,
        # pretrained_text_encoder_name_or_path=pretrained_text_encoder_name_or_path,
        pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path,
        control_frequency=args.ctrl_freq,
    )

    return model


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


# Interpolate the actions to make the robot move smoothly
def interpolate_action(args, prev_action, cur_action):
    steps = np.concatenate((np.array(args.arm_steps_length), np.array(args.arm_steps_length)), axis=0)
    diff = np.abs(cur_action - prev_action)
    step = np.ceil(diff / steps).astype(int)
    step = np.max(step)
    if step <= 1:
        return cur_action[np.newaxis, :]
    new_actions = np.linspace(prev_action, cur_action, step + 1)
    return new_actions[1:]


def get_config(args):
    config = {
        'episode_len': args.max_publish_step,
        'state_dim': 7,
        'chunk_size': args.chunk_size,
        'camera_names': CAMERA_NAMES,
    }
    return config


# Update the observation window buffer
def update_observation_window(config, robot):
    global observation_window
    if observation_window is None:
        observation_window = deque(maxlen=2)

        # Append the first dummy image
        observation_window.append(
            {
                'qpos': None,
                'images':
                    {
                        config["camera_names"][0]: None,
                        config["camera_names"][1]: None,
                    },
            }
        )

    # Get image
    img_front, img_right = frame_queue.get()

    # Get robot state
    q = robot.get_current_joint_q()
    eef = robot.get_current_end()
    eef = 0.0 if eef < 0.05 else 1.0 if eef > 0.95 else eef
    qpos = q + [eef]

    qpos = torch.tensor(qpos, dtype=torch.float).cuda()
    observation_window.append(
        {
            'qpos': qpos,
            'images':
                {
                    config["camera_names"][0]: img_front,
                    config["camera_names"][1]: img_right,
                },
        }
    )


# RDT inference
def inference_fn(config, policy, t):
    global observation_window
    global lang_embeddings

    print(f"Start inference_thread_fn: t={t}")
    while True:
        time1 = time.time()

        # fetch images in sequence [front, right, left]
        image_arrs = [
            observation_window[-2]['images'][config['camera_names'][0]],
            observation_window[-2]['images'][config['camera_names'][1]],

            observation_window[-1]['images'][config['camera_names'][0]],
            observation_window[-1]['images'][config['camera_names'][1]]
        ]

        images = [PImage.fromarray(arr) if arr is not None else None
                  for arr in image_arrs]

        # get last qpos in shape [7, ]
        proprio = observation_window[-1]['qpos']
        # unsqueeze to [1, 7]
        proprio = proprio.unsqueeze(0)

        # actions shaped as [1, 64, 14] in format [left, right]
        actions = policy.step(
            proprio=proprio,
            images=images,
            text_embeds=lang_embeddings
        ).squeeze(0).cpu().numpy()
        # print(f"inference_actions: {actions.squeeze()}")

        print(f"Model inference time: {time.time() - time1} s")

        print(f"Finish inference_thread_fn: t={t}")
        return actions



def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_publish_step', action='store', type=int,
                        help='Maximum number of action publishing steps', default=10000, required=False)
    parser.add_argument('--seed', action='store', type=int,
                        help='Random seed', default=None, required=False)

    parser.add_argument('--img_front_topic', action='store', type=str, help='img_front_topic',
                        default='/camera_f/color/image_raw', required=False)
    parser.add_argument('--img_left_topic', action='store', type=str, help='img_left_topic',
                        default='/camera_l/color/image_raw', required=False)
    parser.add_argument('--img_right_topic', action='store', type=str, help='img_right_topic',
                        default='/camera_r/color/image_raw', required=False)

    parser.add_argument('--img_front_depth_topic', action='store', type=str, help='img_front_depth_topic',
                        default='/camera_f/depth/image_raw', required=False)
    parser.add_argument('--img_left_depth_topic', action='store', type=str, help='img_left_depth_topic',
                        default='/camera_l/depth/image_raw', required=False)
    parser.add_argument('--img_right_depth_topic', action='store', type=str, help='img_right_depth_topic',
                        default='/camera_r/depth/image_raw', required=False)

    parser.add_argument('--puppet_arm_left_cmd_topic', action='store', type=str, help='puppet_arm_left_cmd_topic',
                        default='/master/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_cmd_topic', action='store', type=str, help='puppet_arm_right_cmd_topic',
                        default='/master/joint_right', required=False)
    parser.add_argument('--puppet_arm_left_topic', action='store', type=str, help='puppet_arm_left_topic',
                        default='/puppet/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_topic', action='store', type=str, help='puppet_arm_right_topic',
                        default='/puppet/joint_right', required=False)

    parser.add_argument('--robot_base_topic', action='store', type=str, help='robot_base_topic',
                        default='/odom_raw', required=False)
    parser.add_argument('--robot_base_cmd_topic', action='store', type=str, help='robot_base_topic',
                        default='/cmd_vel', required=False)
    parser.add_argument('--use_robot_base', action='store_true',
                        help='Whether to use the robot base to move around',
                        default=False, required=False)
    parser.add_argument('--publish_rate', action='store', type=int,
                        help='The rate at which to publish the actions',
                        default=30, required=False)
    parser.add_argument('--ctrl_freq', action='store', type=int,
                        help='The control frequency of the robot',
                        default=25, required=False)

    parser.add_argument('--chunk_size', action='store', type=int,
                        help='Action chunk size',
                        default=64, required=False)
    parser.add_argument('--arm_steps_length', action='store', type=float,
                        help='The maximum change allowed for each joint per timestep',
                        default=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2], required=False)

    parser.add_argument('--use_actions_interpolation', action='store_true',
                        help='Whether to interpolate the actions if the difference is too large',
                        default=False, required=False)
    parser.add_argument('--use_depth_image', action='store_true',
                        help='Whether to use depth images',
                        default=False, required=False)

    parser.add_argument('--disable_puppet_arm', action='store_true',
                        help='Whether to disable the puppet arm. This is useful for safely debugging', default=False)

    parser.add_argument('--config_path', type=str, default="configs/base.yaml",
                        help='Path to the config file')
    # parser.add_argument('--cfg_scale', type=float, default=2.0,
    #                     help='the scaling factor used to modify the magnitude of the control features during denoising')
    parser.add_argument('--pretrained_model_name_or_path', type=str, required=True,
                        help='Name or path to the pretrained model')

    parser.add_argument('--lang_embeddings_path', type=str, required=True,
                        help='Path to the pre-encoded language instruction embeddings')

    args = parser.parse_args()
    return args


def camera_thread():
    # Set the record image parameters
    color_image_front = np.zeros((record_img_height, record_img_width, 3), dtype=np.uint8)
    color_image_right = np.zeros((record_img_height, record_img_width, 3), dtype=np.uint8)

    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            # receive the frames
            frames_front = pipeline_front.wait_for_frames()
            color_frame_front = frames_front.get_color_frame()
            frames_right = pipeline_right.wait_for_frames()
            color_frame_right = frames_right.get_color_frame()

            # store the frames only when successfully gotten
            if color_frame_front:
                color_image_front = np.asanyarray(color_frame_front.get_data())
            if color_frame_right:
                color_image_right = np.asanyarray(color_frame_right.get_data())
            if frame_queue.full():
                try:
                    frame_queue.get_nowait() # leave out the old frames
                except queue.Empty:
                    pass
            frame_queue.put((color_image_front, color_image_right)) # only store the newest ones

            # show the frames in the cv2 window
            image_show = np.hstack((color_image_front, color_image_right))
            cv2.imshow('RealSense', image_show)

            # keys detection
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                print('Stopping the cameras ...')
                pipeline_front.stop()
                pipeline_right.stop()
                print('End')
                break

    except Exception as e:
        print(e)
        pipeline_front.stop()
        pipeline_right.stop()
        print('End')


if __name__ == '__main__':
    args = get_arguments()
    if args.seed is not None:
        set_seed(args.seed)
    config = get_config(args)

    # Initialize the realsense cameras
    record_img_width = 640
    record_img_height = 480

    pipeline_front = rs.pipeline()
    config_front = rs.config()
    config_front.enable_device('244622072764')
    config_front.enable_stream(rs.stream.color, record_img_width, record_img_height, rs.format.bgr8, 30)
    pipeline_front.start(config_front)

    pipeline_right = rs.pipeline()
    config_right = rs.config()
    config_right.enable_device('207522077736')
    config_right.enable_stream(rs.stream.color, record_img_width, record_img_height, rs.format.bgr8, 30)
    pipeline_right.start(config_right)

    play_robot = airbot.create_agent(can_interface="can1", end_mode="gripper")

    time.sleep(1)
    print("Cameras and Robot Initialization Succeeded -----")

    frame_queue = queue.Queue(maxsize=1)

    threading.Thread(target=camera_thread, daemon=True).start()


    # Main loop for the manipulation task
    # global lang_embeddings

    # Load rdt model
    policy = make_policy(args)
    print(f"Model successfully loaded from {args.pretrained_model_name_or_path}.")

    lang_dict = torch.load(args.lang_embeddings_path)
    print(f"Running with instruction: \"{lang_dict['instruction']}\" from \"{lang_dict['name']}\"")
    lang_embeddings = lang_dict["embeddings"]

    max_publish_step = config['episode_len']
    chunk_size = config['chunk_size']

    # Initialize position of the airbot
    right0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]
    right1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    play_robot.set_target_end(right0[6], blocking=False)
    play_robot.set_target_joint_q(right0[:6], use_planning=False, blocking=False)
    input("Initialized Pose 0. Press enter to continue")
    play_robot.set_target_end(right1[6], blocking=False)
    play_robot.set_target_joint_q(right1[:6], use_planning=False, blocking=False)
    input("Initialized Pose 1. Press enter to start the task")

    # Initialize the previous action to be the initial robot state
    pre_action = np.zeros(config['state_dim'])
    pre_action[:7] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    action = None

    # Inference loop
    try:
        with torch.inference_mode():
            while True:  ### need to edit
                # The current time step
                t = 0
                action_buffer = np.zeros([chunk_size, config['state_dim']])

                while t < max_publish_step:
                    # Update observation window
                    update_observation_window(config, play_robot)

                    # When coming to the end of the action chunk
                    if t % chunk_size == 0:
                        # Start inference
                        action_buffer = inference_fn(config, policy, t).copy()

                    raw_action = action_buffer[t % chunk_size]
                    action = raw_action
                    # Interpolate the original action sequence
                    if args.use_actions_interpolation:
                        print(f"Time {t}, pre {pre_action}, act {action}")
                        interp_actions = interpolate_action(args, pre_action, action)
                    else:
                        interp_actions = action[np.newaxis, :]
                    # Execute the interpolated actions one by one
                    ctrl_freq = args.ctrl_freq
                    interval = 1.0 / ctrl_freq
                    for act in interp_actions:
                        input("Press Enter to run the next step of action...")
                        start_time = time.time()

                        right_action = act[:7].tolist()
                        if not args.disable_puppet_arm:
                            play_robot.set_target_end(right_action[6], blocking=False)
                            play_robot.set_target_joint_q(right_action[:6], use_planning=False, blocking=False)

                        elapsed = time.time() - start_time
                        sleep_time = interval - elapsed
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                        else:
                            print(f"[Warning] Loop overran. Elapsed time: {elapsed:.2f}s.")
                    t += 1

                    print("Published Step", t)
                    pre_action = action.copy()

    except Exception as e:
        print(e)
        del play_robot
        pipeline_front.stop()
        pipeline_right.stop()
        print('End')