# auto drive test
import numpy as np
import gym
import cv2
from gym import wrappers

import vision
import control

env = gym.make('CarRacing-v2', render_mode='human')
env = wrappers.RecordVideo(env, video_folder='./videos/', name_prefix='pid_video', video_length=6000)

vision_module = vision.BEV_Vision(debug=False)
control_module = control.Car_Controller(debug=False)


def skip_frame(skip_num):
    for _ in range(skip_num):
        env.step((0, 0, 0))


if __name__ == '__main__':
    env.reset()
    action = (0.0, 0.0, 0.0)

    skip_frame(50)

    while True:
        observation, reward, done, info = env.step(action)

        raw_img = np.array(observation, np.uint8)

        velocity, ref_line = vision_module.vision_task(raw_img)

        control_module.feed_back(velocity, ref_line)
        control_module.control_task("MPC")
        # control_module.control_task("PID")
        action = control_module.action

        env.render()
        cv2.waitKey(1)
