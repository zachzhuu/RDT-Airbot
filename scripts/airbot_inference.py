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
# import rospy
import torch
from PIL import Image as PImage
import cv2
import airbot
import pyrealsense2 as rs

from scripts.airbot_model import create_model


# sys.path.append("./")

CAMERA_NAMES = ['cam_high']  # , 'cam_right_wrist', 'cam_left_wrist'

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
    

def get_bot_observation(bot_operator):
    assert isinstance(bot_operator, AirbotOperator)
    # Get proprioceptive data
    q = bot_operator.play_robot.get_current_joint_q()
    eef = bot_operator.play_robot.get_current_end()
    eef = 0.0 if eef < 0.05 else 1.0 if eef > 0.95 else eef
    qpos = q + [eef]
    qpos = np.array(qpos)

    # Get image data
    frames = bot_operator.cam_high_pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        return
    color_image = np.asanyarray(color_frame.get_data())
    color_image = color_image.astype(np.uint8)  # color format aligned with training

    return (color_image, qpos)


# Update the observation window buffer
def update_observation_window(config, bot_operator):
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
                        # config["camera_names"][1]: None,
                        # config["camera_names"][2]: None,
                    },
            }
        )
    
    img_front, puppet_arm_right = get_bot_observation(bot_operator)
    
    qpos = torch.from_numpy(puppet_arm_right).float().cuda()
    observation_window.append(
        {
            'qpos': qpos,
            'images':
                {
                    config["camera_names"][0]: img_front,
                    # config["camera_names"][1]: img_right,
                    # config["camera_names"][2]: img_left,
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
            # observation_window[-2]['images'][config['camera_names'][1]],
            # observation_window[-2]['images'][config['camera_names'][2]],
            
            observation_window[-1]['images'][config['camera_names'][0]]
            # observation_window[-1]['images'][config['camera_names'][1]],
            # observation_window[-1]['images'][config['camera_names'][2]]
        ]
        
        # fetch debug images in sequence [front, right, left]
        # image_arrs = [
        #     preload_images[config['camera_names'][0]][max(t - 1, 0)],
        #     preload_images[config['camera_names'][2]][max(t - 1, 0)],
        #     preload_images[config['camera_names'][1]][max(t - 1, 0)],
        #     preload_images[config['camera_names'][0]][t],
        #     preload_images[config['camera_names'][2]][t],
        #     preload_images[config['camera_names'][1]][t]
        # ]
        # # encode the images
        # for i in range(len(image_arrs)):
        #     image_arrs[i] = cv2.imdecode(np.frombuffer(image_arrs[i], np.uint8), cv2.IMREAD_COLOR)
        # proprio = torch.from_numpy(preload_images['qpos'][t]).float().cuda()
        
        images = [PImage.fromarray(arr) if arr is not None else None
                  for arr in image_arrs]
        
        # for i, pos in enumerate(['f', 'r', 'l'] * 2):
        #     images[i].save(f'{t}-{i}-{pos}.png')
        
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
        
        # print(f"Finish inference_thread_fn: t={t}")
        return actions


# Main loop for the manipulation task
def model_inference(args, config, bot_operator):
    global lang_embeddings

    assert isinstance(bot_operator, AirbotOperator)
    # Load rdt model
    policy = make_policy(args)
    
    lang_dict = torch.load(args.lang_embeddings_path)
    print(f"Running with instruction: \"{lang_dict['instruction']}\" from \"{lang_dict['name']}\"")
    lang_embeddings = lang_dict["embeddings"]
    
    max_publish_step = config['episode_len']
    chunk_size = config['chunk_size']

    # TODO: check the initial pose of our robot arm
    # Initialize position of the puppet arm
    # left0 = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, 3.557830810546875]
    right0 = [0.32948363, -0.10769133, 0.18022917, -0.17248239, 0.5107854, 0.2793215, 0.79456127, 1.0]
    # left1 = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, -0.3393220901489258]
    right1 = [0.32948363, -0.10769133, 0.18022917, -0.17248239, 0.5107854, 0.2793215, 0.79456127, 0.0]
    
    bot_operator.playbot_publish_continuous(right0)
    input("Press enter to continue")
    bot_operator.playbot_publish_continuous(right1)

    # Initialize the previous action to be the initial robot state
    pre_action = np.zeros(config['state_dim'])
    pre_action[:7] = np.array(
        [0.32948363, -0.10769133, 0.18022917, -0.17248239, 0.5107854, 0.2793215, 0.79456127, 0.0]
    )
    action = None

    # Inference loop
    with torch.inference_mode():
        while True:
            # The current time step
            t = 0
            # rate = rospy.Rate(args.publish_rate)
    
            action_buffer = np.zeros([chunk_size, config['state_dim']])
            
            while t < max_publish_step:
                # Update observation window
                # TODO: mark
                update_observation_window(config, bot_operator)
                
                # When coming to the end of the action chunk
                if t % chunk_size == 0:
                    # Start inference
                    action_buffer = inference_fn(config, policy, t).copy()
                
                raw_action = action_buffer[t % chunk_size]
                action = raw_action
                # Interpolate the original action sequence
                if args.use_actions_interpolation:
                    # print(f"Time {t}, pre {pre_action}, act {action}")
                    interp_actions = interpolate_action(args, pre_action, action)
                else:
                    interp_actions = action[np.newaxis, :]
                # Execute the interpolated actions one by one
                for act in interp_actions:
                    ctrl_freq = args.ctrl_freq
                    interval = 1.0 / ctrl_freq

                    start_time = time.time()

                    right_action = act[:7]
                    # right_action = act[7:14]
                    if not args.disable_puppet_arm:
                        bot_operator.playbot_publish(right_action)
                    
                    elapsed = time.time() - start_time
                    sleep_time = interval - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    else:
                        print(f"[Warning] Loop overran. Elapsed time: {elapsed:.2f}s, ")
                t += 1
                
                print("Published Step", t)
                pre_action = action.copy()


# ROS operator class
class AirbotOperator:
    def __init__(self, args):
        self.robot_base_deque = None
        self.puppet_arm_right_deque = None
        self.puppet_arm_left_deque = None
        self.img_front_deque = None
        self.img_right_deque = None
        self.img_left_deque = None
        self.img_front_depth_deque = None
        self.img_right_depth_deque = None
        self.img_left_depth_deque = None
        self.bridge = None
        self.puppet_arm_left_publisher = None
        self.puppet_arm_right_publisher = None
        self.robot_base_publisher = None
        self.puppet_arm_publish_thread = None
        self.puppet_arm_publish_lock = None
        self.args = args
        self.airbot_vel = 0.08
        self.init()
        # self.init_ros()
        self.init_bot()

    def init(self):
        self.img_left_deque = deque()
        self.img_right_deque = deque()
        self.img_front_deque = deque()
        self.img_left_depth_deque = deque()
        self.img_right_depth_deque = deque()
        self.img_front_depth_deque = deque()
        self.puppet_arm_left_deque = deque()
        self.puppet_arm_right_deque = deque()
        self.robot_base_deque = deque()
        self.puppet_arm_publish_lock = threading.Lock()
        self.puppet_arm_publish_lock.acquire()

    def playbot_publish(self, right):
        self.play_robot.set_target_end(right[-1], blocking=False)
        self.play_robot.set_target_pose([right[:3], right[3:7]], use_planning=False, blocking=False, vel=self.airbot_vel)

    def playbot_publish_continuous(self, right):
        self.play_robot.set_target_end(right[-1], blocking=False)
        self.play_robot.set_target_pose([right[:3], right[3:7]], use_planning=True, blocking=False, vel=self.airbot_vel)

    def init_bot(self):
        self.play_robot = airbot.create_agent(can_interface="can1", end_mode="gripper")
        self.teacher_robot = None

        self.joint_limits = {
                    'lower_bounds': np.array([-177, -167, -2, -145, -97, -176]),
                    'upper_bounds': np.array([117, 7, 177, 145, 97, 176])
                }
        
        # Init Realsense Camera
        self.cam_high_pipeline = rs.pipeline()
        cam_high_config = rs.config()
        # cam_high_config.enable_stream(
        #     stream_type=rs.stream.depth, 
        #     width=640, 
        #     height=480, 
        #     format=rs.format.z16,
        #     framerate=30
        # )
        cam_high_config.enable_stream(
            stream_type=rs.stream.color,
            width=640,
            height=480,
            format=rs.format.bgr8,
            framerate=30
        )
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)

        # Start images streaming
        profile = self.cam_high_pipeline.start(cam_high_config)
        device = profile.get_device()
        device.hardware_reset()
        # profile = pipeline.get_active_profile()
        _ = profile.get_stream(rs.stream.color)


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
    # TODO: set the arm_steps_length for Airbot
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
    args = get_arguments()
    ros_operator = AirbotOperator(args)
    if args.seed is not None:
        set_seed(args.seed)
    config = get_config(args)
    model_inference(args, config, ros_operator)
    print("Loop finished.")


if __name__ == '__main__':
    main()
