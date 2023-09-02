# auto drive test
import numpy as np
import gym
import cv2
from gym import wrappers

import vision
import control
import data_recorder

env = gym.make('CarRacing-v2', render_mode='human')  # , render_mode='human'
env = wrappers.RecordVideo(env, video_folder='./videos/', name_prefix='pid_video', video_length=6000)

vision_module = vision.BEV_Vision(debug=False)
control_module = control.Car_Controller(debug=False)
# video = data_recorder.VideoRecorder("./videos/mpc.avi")


def skip_frame(skip_num):
    for _ in range(skip_num):
        env.step((0, 0, 0))


if __name__ == '__main__':
    env.reset()
    action = (0.0, 0.2, 0)

    skip_frame(50)

    ct = 0

    while True:
        observation, reward, done, info = env.step(action)

        raw_img = np.array(observation, np.uint8)

        # vision_module.get_raw_image(raw_img)
        # vision_module.lane_detector()
        # vision_module.reference_line_extractor()
        # vision_module.optical_flow()

        velocity, ref_line = vision_module.vision_task(raw_img)

        control_module.feed_back(velocity, ref_line)
        control_module.velocity_controller(0.8)
        control_module.direction_controller()
        # control_module.MPC_solve()

        # env.render()

        # if ct < 1000:
        #     video.record_frame()
        #     ct += 1
        # elif ct == 1000:
        #     video.stop()
        #     ct += 1

        action = control_module.action

        cv2.waitKey(1)
