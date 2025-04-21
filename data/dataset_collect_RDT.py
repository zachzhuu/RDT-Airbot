import airbot
import os
import time
import pyrealsense2 as rs
import numpy as np
import cv2
import sys
import pickle
import multiprocessing
from termcolor import cprint
import threading
import queue
import copy

def record_data(color_image, depth_image, play_robot, teacher_robot):
    global record_path, data, record_episode, record_flag, record_step, record_steps_num

    # record data: play_robot -> state, teacher_robot -> action
    data['timestamp'][record_step] = time.time()
    data['robot_arm_joint'][record_step] = np.array(play_robot.get_current_joint_q())
    data['robot_gripper_joint'][record_step] = np.array(play_robot.get_current_end())
    data['robot_arm_joint_v'][record_step] = np.array(play_robot.get_current_joint_v())
    data['robot_end_effector_position'][record_step] = np.array(play_robot.get_current_translation())
    data['robot_end_effector_rotation'][record_step] = np.array(play_robot.get_current_rotation())
    data['teacher_arm_joint'][record_step] = np.array(teacher_robot.get_current_joint_q())
    data['teacher_gripper_joint'][record_step] = np.array(teacher_robot.get_current_end())
    data['teacher_arm_joint_v'][record_step] = np.array(teacher_robot.get_current_joint_v())
    data['teacher_end_effector_position'][record_step] = np.array(teacher_robot.get_current_translation())
    data['teacher_end_effector_rotation'][record_step] = np.array(teacher_robot.get_current_rotation())
    data['color_image'][record_step] = color_image
    data['depth_image'][record_step] = depth_image

    # control the play_robot
    play_robot.set_target_end(data['teacher_gripper_joint'][record_step], blocking=False)
    play_robot.set_target_joint_q(data['teacher_arm_joint'][record_step], use_planning=False, blocking=False)

    record_step += 1
    # save the data and initialize the data
    if record_step >= record_steps_num:
        record_flag = False
        filename = os.path.join(record_path, f'ep{record_episode:02}.pkl')
        data_copy = copy.deepcopy(data)
        data_saving = multiprocessing.Process(target=save_data, args=(filename, data_copy))  # this step will take long time. so we run it in subprocess
        data_saving.start()

        record_step = 0
        record_episode += 1
        data = {
            'timestamp': np.zeros([record_steps_num]),
            'robot_arm_joint': np.zeros([record_steps_num, 6]),
            'robot_gripper_joint': np.zeros([record_steps_num]),
            'robot_arm_joint_v': np.zeros([record_steps_num, 6]),
            'robot_end_effector_position': np.zeros([record_steps_num, 3]),
            'robot_end_effector_rotation': np.zeros([record_steps_num, 4]),
            'teacher_arm_joint': np.zeros([record_steps_num, 6]),
            'teacher_gripper_joint': np.zeros([record_steps_num]),
            'teacher_arm_joint_v': np.zeros([record_steps_num, 6]),
            'teacher_end_effector_position': np.zeros([record_steps_num, 3]),
            'teacher_end_effector_rotation': np.zeros([record_steps_num, 4]),
            'color_image': np.zeros([record_steps_num, record_img_height, record_img_width, 3]),
            'depth_image': np.zeros([record_steps_num, record_img_height, record_img_width]),
        }


def save_data(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
        print(f"Data saved to {filename}")


def record_loop():
    global record_flag, record_freq, play_robot, teacher_robot
    period = 1.0 / record_freq
    next_time = time.time()

    while record_flag:
        color_image, depth_image = frame_queue.get(timeout=period)
        record_data(color_image, depth_image, play_robot, teacher_robot)
        next_time += period
        sleep_duration = next_time - time.time()

        if sleep_duration > 0:
            time.sleep(sleep_duration)
        else:
            print("Warning: Processing slower than real-time.")
            next_time = time.time()  # reset time to avoid drift


if __name__ == '__main__':
    record_path = 'data/raw_test'
    # if os.path.exists(record_path) and the directory is not empty, ask the user if they want to overwrite
    if os.path.exists(record_path) and os.listdir(record_path):
        cprint('Data already exists at {}'.format(record_path), 'red')
        cprint("If you want to overwrite, delete the existing directory first.", "red")
        cprint("Do you want to overwrite? (y/n)", "red")
        user_input = input()
        if user_input == 'y':
            cprint('Overwriting {}'.format(record_path), 'red')
            os.system('rm -rf {}'.format(record_path))
        else:
            cprint('Exiting', 'red')
            sys.exit()
    # if the directory does not exist, create it
    elif not os.path.exists(record_path):
        os.mkdir(record_path)
        print('Data will be saved to {}'.format(record_path))
    else:
        print('Data will be saved to {}'.format(record_path))
    os.makedirs(record_path, exist_ok=True)

    # Init realsense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)

    # Start images streaming
    profile = pipeline.start(config)
    device = profile.get_device()
    device.hardware_reset()
    depth_stream = profile.get_stream(rs.stream.depth)

    # Store the camera data
    camera_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
    camera_data = {
        'intrinsics_matrix': [
            [camera_intrinsics.fx, 0, camera_intrinsics.ppx],
            [0, camera_intrinsics.fy, camera_intrinsics.ppy],
            [0, 0, 1]
        ],
        'camera_params': {
            'width': camera_intrinsics.width,
            'height': camera_intrinsics.height,
            'fx': camera_intrinsics.fx,
            'fy': camera_intrinsics.fy,
            'cx': camera_intrinsics.ppx,
            'cy': camera_intrinsics.ppy,
        }
    }
    filename = os.path.join(record_path, f'camera_data.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(camera_data, f)
        print(f"Camera data has been saved to {filename}")

    # Init robot
    teacher_robot = airbot.create_agent(can_interface="can0", forearm_type="encoder", bigarm_type='encoder', end_mode="encoder")
    play_robot = airbot.create_agent(can_interface="can1", end_mode="gripper")

    # Record variables
    record_episode = 0
    record_flag = False
    record_step = 0
    record_freq = 10
    record_time = 30
    record_steps_num = record_freq * record_time
    record_img_width = 640
    record_img_height = 480
    data = {
        'timestamp': np.zeros([record_steps_num]),
        'robot_arm_joint': np.zeros([record_steps_num, 6]),
        'robot_gripper_joint': np.zeros([record_steps_num]),
        'robot_arm_joint_v': np.zeros([record_steps_num, 6]),
        'robot_end_effector_position': np.zeros([record_steps_num, 3]),
        'robot_end_effector_rotation': np.zeros([record_steps_num, 4]),
        'teacher_arm_joint': np.zeros([record_steps_num, 6]),
        'teacher_gripper_joint': np.zeros([record_steps_num]),
        'teacher_arm_joint_v': np.zeros([record_steps_num, 6]),
        'teacher_end_effector_position': np.zeros([record_steps_num, 3]),
        'teacher_end_effector_rotation': np.zeros([record_steps_num, 4]),
        'color_image': np.zeros([record_steps_num, record_img_height, record_img_width, 3]),
        'depth_image': np.zeros([record_steps_num, record_img_height, record_img_width]),
    }

    record_para = {
        'record_freq': record_freq,
        'record_time': record_time
    }
    filename = os.path.join(record_path, f'record_para.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(record_para, f)
        print(f"Record parameters have been saved to {filename}")

    time.sleep(1)
    print("Start")

    try:
        frame_queue = queue.Queue(maxsize=1)
        while True:
            # receive frames and show
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            # 扔掉旧帧，只保留最新帧
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()  # 丢掉旧帧
                except queue.Empty:
                    pass
            frame_queue.put((color_image, depth_image))

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            image_show = np.hstack((color_image, depth_colormap))
            cv2.putText(image_show, f'epoch={record_episode}', (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(image_show, (320, 240), 5, (255, 0, 0), -1)
            if record_flag == True:
                cv2.circle(image_show, (30, 60), 20, (0, 0, 255), -1)
                cv2.putText(image_show, str(record_step), (30, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow('RealSense', image_show)

            # keys detection
            key = cv2.waitKey(1)
            if key & 0xFF == 32:  # key: space
                if not record_flag:
                    record_flag = True
                    threading.Thread(target=record_loop, daemon=True).start()
            if key & 0xFF == ord('d'):
                record_episode -= 1
                print(f'Delete previous dataset, next epoch is {record_episode}')
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                del play_robot
                del teacher_robot
                pipeline.stop()
                print('\r\nEnd')
                break

    except Exception as e:
        print(e)
        del play_robot
        pipeline.stop()
        print('\r\nEnd')
