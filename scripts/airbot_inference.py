#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""

import argparse
import time
import yaml
from collections import deque
import traceback
import copy

import numpy as np
# import rospy
import torch
from PIL import Image as PImage
import airbot
import pyrealsense2 as rs

from scripts.airbot_model import create_model


# sys.path.append("./")

CAMERA_NAMES = ['cam_high', 'cam_right_wrist']  # , 'cam_left_wrist'
# Initialize the realsense cameras
record_img_width = 640
record_img_height = 480

observation_window = None

lang_embeddings = None

# debug
preload_images = None

# Initial pose of the robot arm
right0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
right1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]

# Initialize the model
def make_policy(args):
    print(f"Loading model from {args.pretrained_model_name_or_path}...")
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
    print(f"Model successfully loaded from {args.pretrained_model_name_or_path}.")
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
    global pipeline_front
    global pipeline_right
    
    if observation_window is None:
        observation_window = deque(maxlen=2)

        # Append the first dummy image
        observation_window.append(
            {
                'qpos': None,
                'images':
                    {
                        config["camera_names"][0]: None,
                        config["camera_names"][1]: None
                    },
            }
        )

    # Get images
    frames_front = pipeline_front.wait_for_frames()
    frames_right = pipeline_right.wait_for_frames()
    color_frame_front = frames_front.get_color_frame()
    color_frame_right = frames_right.get_color_frame()
    
    if not color_frame_front or not color_frame_right:
        return
    img_front = np.asanyarray(color_frame_front.get_data())
    img_right = np.asanyarray(color_frame_right.get_data())
    
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
                    config["camera_names"][1]: img_right
                },
        }
    )


# RDT inference
def inference_fn(config, policy, t):
    global observation_window
    global lang_embeddings
    
    print(f"Start inference: t={t}")
    time1 = time.time()     

    # fetch images in sequence [front, right, left]
    image_arrs = [
        observation_window[-2]['images'][config['camera_names'][0]],
        observation_window[-2]['images'][config['camera_names'][1]],
        # observation_window[-2]['images'][config['camera_names'][2]],
        
        observation_window[-1]['images'][config['camera_names'][0]],
        observation_window[-1]['images'][config['camera_names'][1]]
        # observation_window[-1]['images'][config['camera_names'][2]]
    ]
    
    images = [PImage.fromarray(arr) if arr is not None else None
                for arr in image_arrs]
    
    # get last qpos in shape [7, ]
    proprio = observation_window[-1]['qpos']
    # unsqueeze to [1, 7]
    proprio = proprio.unsqueeze(0)
    
    # actions shaped as [1, 64, 7]
    actions = policy.step(
        proprio=proprio,
        images=images,
        text_embeds=lang_embeddings 
    ).squeeze(0).cpu().numpy()
    # print(f"inference_actions: {actions.squeeze()}")
    
    print(f"Model inference time: {time.time() - time1} s")
    
    # print(f"Finish inference_thread_fn: t={t}")
    return actions


# Main loop for the manipulation task
def model_inference(args, config, play_robot):
    global lang_embeddings
    
    # Load rdt model
    policy = make_policy(args)
    
    lang_dict = torch.load(args.lang_embeddings_path)
    print(f"Loaded language embeddings from {args.lang_embeddings_path}")
    # print(f"Running with instruction: \"{lang_dict['instruction']}\" from \"{lang_dict['name']}\"")
    lang_embeddings = lang_dict  # ["embeddings"]
    
    max_publish_step = config['episode_len']
    chunk_size = config['chunk_size']
    
    # Back to the initial position
    play_robot.set_target_end(right1[-1], blocking=False)
    play_robot.set_target_joint_q(right1[:6], use_planning=False, blocking=False)
    input("Press enter to the initial pose...")
    play_robot.set_target_end(right0[-1], blocking=False)
    play_robot.set_target_joint_q(right0[:6], use_planning=False, blocking=False)
    input("Press enter to start the task...")

    # Initialize the previous action to be the initial robot state
    pre_action = np.zeros(config['state_dim'])
    pre_action[:7] = np.array(
        copy.deepcopy(right0)
    )
    action = None
    
    # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)

    # Inference loop
    with torch.inference_mode():
        # The current time step
        t = 0
        action_buffer = np.zeros([chunk_size, config['state_dim']])
        
        print("Start the inference loop...")
        while t < max_publish_step:
            # TODO: add a key interruption w/o cv2
            # Update observation window
            update_observation_window(config, play_robot)
            # image_to_show = update_observation_window(config, play_robot)
            # cv2.imshow('RealSense', image_to_show)
            
            # When coming to the end of the action chunk
            if t % chunk_size == 0:
                # Start inference
                action_buffer = inference_fn(config, policy, t).copy()
            
            raw_action = action_buffer[t % chunk_size]
            # Interpolate the original action sequence
            if args.use_actions_interpolation:
                # print(f"Time {t}, pre {pre_action}, act {action}")
                interp_actions = interpolate_action(args, pre_action, raw_action)
            else:
                interp_actions = raw_action[np.newaxis, :]
            # Execute the interpolated actions one by one
            for act in interp_actions:
                ctrl_freq = args.ctrl_freq
                interval = 1.0 / ctrl_freq

                start_time = time.time()

                right_action = act[:7].tolist()
                # right_action = act[7:14]
                if not args.disable_puppet_arm:
                    play_robot.set_target_end(right_action[-1], blocking=False)
                    play_robot.set_target_joint_q(right_action[:6], use_planning=False, blocking=False)
                
                elapsed = time.time() - start_time
                sleep_time = interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    print(f"[Warning] Loop overran. Elapsed time: {elapsed:.2f}s.")
            t += 1
            
            print(f"Published step: {t}")
            pre_action = action.copy()
    
    print("Loop finished.")
    input("Press enter to delete robot...")
    play_robot.set_target_end(right0[-1], blocking=False)
    play_robot.set_target_joint_q(right0[:6], use_planning=False, blocking=False)
    time.sleep(1) 
    del play_robot
    pipeline_front.stop()
    pipeline_right.stop()
    print("Play robot successfully deleted.")


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_publish_step', action='store', type=int, 
                        help='Maximum number of action publishing steps', default=10000, required=False)
    parser.add_argument('--seed', action='store', type=int, 
                        help='Random seed', default=None, required=False)
    
    parser.add_argument('--ctrl_freq', action='store', type=int, 
                        help='The control frequency of the robot',
                        default=25, required=False)
    
    parser.add_argument('--chunk_size', action='store', type=int, 
                        help='Action chunk size',
                        default=64, required=False)
    # TODO: set the arm_steps_length for Airbot
    parser.add_argument('--arm_steps_length', action='store', type=float, 
                        help='The maximum change allowed for each joint per timestep',
                        default=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2], required=False)

    parser.add_argument('--use_actions_interpolation', action='store_true',
                        help='Whether to interpolate the actions if the difference is too large',
                        default=False, required=False)
    
    parser.add_argument('--disable_puppet_arm', action='store_true',
                        help='Whether to disable the puppet arm. This is useful for safely debugging',default=False)
    
    parser.add_argument('--config_path', type=str, default="configs/base.yaml", 
                        help='Path to the config file')
    # parser.add_argument('--cfg_scale', type=float, default=2.0,
    #                     help='the scaling factor used to modify the magnitude of the control features during denoising')
    parser.add_argument('--pretrained_model_name_or_path', type=str, required=True, help='Name or path to the pretrained model')
    
    parser.add_argument('--lang_embeddings_path', type=str, required=True, 
                        help='Path to the pre-encoded language instruction embeddings')
    
    args = parser.parse_args()
    return args


def main():
    global pipeline_front
    global pipeline_right
    
    args = get_arguments()
    
    # Initialize the bot
    play_robot = airbot.create_agent(can_interface="can1", end_mode="gripper")
    teacher_robot = None
    
    # Init Realsense Camera
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

    print("Airbot operator and cameras successfully initialized.")

    if args.seed is not None:
        set_seed(args.seed)
    config = get_config(args)
    
    try:
        ###### Main Loop of the Task ######
        model_inference(args, config, play_robot)
        ###### End of the Task Loop ######
    except Exception as e:
        print(f"Exception occurred during model inference: {e}")
        play_robot.set_target_end(right0[-1], blocking=False)
        play_robot.set_target_joint_q(right0[:6], use_planning=False, blocking=False)
        time.sleep(1) 
        del play_robot
        pipeline_front.stop()
        pipeline_right.stop()
        traceback.print_exc()
        

if __name__ == '__main__':
    main()
    