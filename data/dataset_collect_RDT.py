import airbot
import os
import time
import pyrealsense2 as rs
import numpy as np
import cv2
import sys
import pickle
from termcolor import cprint
import threading
import queue
import imageio
import concurrent.futures


def record_robot_data(play_robot, teacher_robot):
    '''The function is the main function of data reading and recording.'''
    global data, record_flag, record_step, record_steps_num, control_in_main

    # record data: play_robot -> state, teacher_robot -> action
    data['timestamp'][record_step] = time.time()
    data['robot_arm_joint'][record_step] = play_robot.get_current_joint_q()
    data['robot_gripper_joint'][record_step] = play_robot.get_current_end()
    data['teacher_arm_joint'][record_step] = teacher_robot.get_current_joint_q()
    data['teacher_gripper_joint'][record_step] = teacher_robot.get_current_end()

    # control the play_robot
    play_robot.set_target_end(data['teacher_gripper_joint'][record_step], blocking=False)
    play_robot.set_target_joint_q(data['teacher_arm_joint'][record_step], use_planning=False, blocking=False)

    record_step += 1
    # initialize the parameters
    if record_step >= record_steps_num:
        record_flag = False
        control_in_main = True
        record_step = 0


def save_single_frame(record_dir, step_idx, color_image_e, depth_image_e, color_image_w):
    rgb_ex_dir = os.path.join(record_dir, "rgb_ex")
    depth_ex_dir = os.path.join(record_dir, "depth_ex")
    rgb_wrist_dir = os.path.join(record_dir, "rgb_wrist")

    os.makedirs(rgb_ex_dir, exist_ok=True)
    os.makedirs(depth_ex_dir, exist_ok=True)
    os.makedirs(rgb_wrist_dir, exist_ok=True)

    rgb_image_e = cv2.cvtColor(color_image_e, cv2.COLOR_BGR2RGB)
    rgb_image_w = cv2.cvtColor(color_image_w, cv2.COLOR_BGR2RGB)

    imageio.imwrite(os.path.join(rgb_ex_dir, f"{step_idx:04d}.png"), rgb_image_e)
    imageio.imwrite(os.path.join(depth_ex_dir, f"{step_idx:04d}.png"), depth_image_e)
    imageio.imwrite(os.path.join(rgb_wrist_dir, f"{step_idx:04d}.png"), rgb_image_w)


def record_loop():
    global record_path, record_flag, record_freq, record_episode, record_step, play_robot, teacher_robot, data

    # 初始化线程池
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        period = 1.0 / record_freq
        next_time = time.time()
        save_dir = os.path.join(record_path, f'ep{record_episode:03}')
        while record_flag:
            try:
                color_image_e, depth_image_e, color_image_w = frame_queue.get(timeout=period)
            except queue.Empty:
                print(f"Warning: Frame queue is empty in step {record_step}")

            # 使用线程池提交保存任务
            executor.submit(save_single_frame, save_dir, record_step,
                            color_image_e.copy(), depth_image_e.copy(), color_image_w.copy())
            record_robot_data(play_robot, teacher_robot)

            next_time += period
            sleep_duration = next_time - time.time()

            if sleep_duration > 0:
                time.sleep(sleep_duration)
            else:
                print(f"Warning: Processing slower than real-time in step {record_step}.")
                next_time = time.time()  # reset time to avoid drift

        # Save meta data in the main thread after recording finishes
        meta_save_path = os.path.join(save_dir, 'meta.npz')
        np.savez_compressed(meta_save_path,
                            record_time=data['record_time'],
                            record_freq=data['record_freq'],
                            timestamp=np.array(data['timestamp']).reshape(-1, 1),
                            robot_arm_joint=np.array(data['robot_arm_joint']),
                            robot_gripper_joint=np.array(data['robot_gripper_joint']).reshape(-1, 1),
                            teacher_arm_joint=np.array(data['teacher_arm_joint']),
                            teacher_gripper_joint=np.array(data['teacher_gripper_joint']).reshape(-1, 1))
        print(f"Meta data saved to {meta_save_path}")

if __name__ == '__main__':
    record_path = 'RDT_test'
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

    # Set the record image parameters
    record_img_width = 640
    record_img_height = 480
    color_image_ex = np.zeros((record_img_height, record_img_width, 3), dtype=np.uint8)
    depth_image_ex = np.zeros((record_img_height, record_img_width), dtype=np.uint16)
    color_image_wrist = np.zeros((record_img_height, record_img_width, 3), dtype=np.uint8)

    # Initialize the realsense cameras
    pipeline_ex = rs.pipeline()
    config_ex = rs.config()
    config_ex.enable_device('244622072764')
    config_ex.enable_stream(rs.stream.depth, record_img_width, record_img_height, rs.format.z16, 30)
    config_ex.enable_stream(rs.stream.color, record_img_width, record_img_height, rs.format.bgr8, 30)
    profile_ex = pipeline_ex.start(config_ex)

    pipeline_wrist = rs.pipeline()
    config_wrist = rs.config()
    config_wrist.enable_device('207522077736')
    config_wrist.enable_stream(rs.stream.color, record_img_width, record_img_height, rs.format.bgr8, 30)
    profile_wrist = pipeline_wrist.start(config_wrist)

    # Store the camera data
    depth_stream_ex = profile_ex.get_stream(rs.stream.depth)
    camera_intrinsics_ex = depth_stream_ex.as_video_stream_profile().get_intrinsics()
    camera_data_ex = {
        'intrinsics_matrix': [
            [camera_intrinsics_ex.fx, 0, camera_intrinsics_ex.ppx],
            [0, camera_intrinsics_ex.fy, camera_intrinsics_ex.ppy],
            [0, 0, 1]
        ],
        'camera_params': {
            'width': camera_intrinsics_ex.width,
            'height': camera_intrinsics_ex.height,
            'fx': camera_intrinsics_ex.fx,
            'fy': camera_intrinsics_ex.fy,
            'cx': camera_intrinsics_ex.ppx,
            'cy': camera_intrinsics_ex.ppy,
        }
    }
    filename = os.path.join(record_path, f'camera_data.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(camera_data_ex, f)
        print(f"Camera data has been saved to {filename}")

    # Initialize the robots
    teacher_robot = airbot.create_agent(can_interface="can0", forearm_type="encoder", bigarm_type='encoder', end_mode="encoder")
    play_robot = airbot.create_agent(can_interface="can1", end_mode="gripper")

    # Initialize the record variables
    record_episode = -1
    record_step = 0
    record_flag = False
    control_in_main = True
    record_freq = 25
    record_time = 30
    record_steps_num = record_freq * record_time

    time.sleep(1)
    print("Start----------")

    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)

    try:
        frame_queue = queue.Queue(maxsize=1)
        while True:
            # receive the frames
            frames_ex = pipeline_ex.wait_for_frames()
            color_frame_ex = frames_ex.get_color_frame()
            depth_frame_ex = frames_ex.get_depth_frame()
            frames_wrist = pipeline_wrist.wait_for_frames()
            color_frame_wrist = frames_wrist.get_color_frame()

            # store the frames only when successfully gotten
            if color_frame_ex:
                color_image_ex = np.asanyarray(color_frame_ex.get_data())
            if depth_frame_ex:
                depth_image_ex = np.asanyarray(depth_frame_ex.get_data())
            if color_frame_wrist:
                color_image_wrist = np.asanyarray(color_frame_wrist.get_data())
            if frame_queue.full():
                try:
                    frame_queue.get_nowait() # leave out the old frames
                except queue.Empty:
                    pass
            frame_queue.put((color_image_ex, depth_image_ex, color_image_wrist)) # only store the newest ones

            # show the frames in the cv2 window
            depth_colormap_ex = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_ex, alpha=0.03), cv2.COLORMAP_JET)
            text_info_canvas = np.zeros_like(color_image_wrist)
            cv2.putText(text_info_canvas, f'Frequency: {record_freq}, Time: {record_time}', (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(text_info_canvas, f'Episode: {record_episode}', (150, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(text_info_canvas, f'Step: {record_step}/{record_steps_num}', (150, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            top_row = np.hstack((color_image_ex, depth_colormap_ex))
            bottom_row = np.hstack((color_image_wrist, text_info_canvas))
            image_show = np.vstack((top_row, bottom_row))
            cv2.imshow('RealSense', image_show)

            # control the play_robot in main function
            if control_in_main:
                target_pos = teacher_robot.get_current_joint_q()
                target_end = teacher_robot.get_current_end()
                play_robot.set_target_end(target_end, blocking=False)
                play_robot.set_target_joint_q(target_pos, use_planning=False, blocking=False)

            # keys detection
            key = cv2.waitKey(1)
            if key & 0xFF == 32:  # key: space
                if not record_flag:
                    control_in_main = False
                    record_flag = True
                    record_episode += 1
                    data = {
                        'record_freq': record_freq,
                        'record_time': record_time,
                        'timestamp': [0.0] * record_steps_num,
                        'robot_arm_joint': [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * record_steps_num,
                        'robot_gripper_joint': [0.0] * record_steps_num,
                        'teacher_arm_joint': [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * record_steps_num,
                        'teacher_gripper_joint': [0.0] * record_steps_num,
                        'color_image_ex': np.zeros([record_steps_num, record_img_height, record_img_width, 3]),
                        'depth_image_ex': np.zeros([record_steps_num, record_img_height, record_img_width]),
                        'color_image_wrist': np.zeros([record_steps_num, record_img_height, record_img_width, 3])
                    }
                    threading.Thread(target=record_loop, daemon=True).start()
            if key & 0xFF == ord('d'):
                record_episode -= 1
                print(f'Delete previous dataset, next epoch is {record_episode}')
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                print('Deleting the robot ...')
                del play_robot
                del teacher_robot
                pipeline_ex.stop()
                pipeline_wrist.stop()
                print('End')
                break

    except Exception as e:
        print(e)
        del play_robot
        del teacher_robot
        pipeline_ex.stop()
        pipeline_wrist.stop()
        print('End')
