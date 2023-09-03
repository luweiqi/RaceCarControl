import numpy as np
import math
from scipy import spatial
from scipy import optimize
import cv2


class Car_Controller:
    def __init__(self, debug=True):
        self.debug = debug
        self.velocity = None
        self.ref_line = None

        self.ref_kd_tree = None

        self.vel_pd = [0.6, 0.5]
        self.action = [0, 0, 0]

        # MPC parameter
        self.forward_step = 4

    def feed_back(self, velocity, ref_line):
        self.velocity = velocity
        self.ref_line = ref_line

        if len(self.ref_line) != 0:
            self.ref_kd_tree = spatial.cKDTree(np.array(self.ref_line))

        if self.debug:
            print("******** velocity ********")
            print(self.velocity)
            print("                          ")

    def velocity_controller(self, ref_v):
        vel = math.sqrt(self.velocity[0] ** 2 + self.velocity[1] ** 2)
        vc = self.vel_pd[0] * (ref_v - vel)
        vc = min(1.0, vc)
        if vc >= 0:
            self.action[1] = vc
            self.action[2] = 0
        else:
            vc = max(-1.0, vc)
            self.action[1] = 0
            self.action[2] = -vc

    def direction_controller(self):
        foresee = min(len(self.ref_line), 10)
        if foresee == 2:
            while True:
                pass

        if foresee != 0:
            c0 = 0.1 * (self.ref_line[foresee - 1][1] - 48)
            c0 = max(-1.0, c0)
            c0 = min(1.0, c0)
            self.action[0] = c0

    @staticmethod
    def vehicle_kinematics_model(current_x, current_y, current_yaw, velocity, steer_angle):
        dt = 4.0
        car_l = 4.0

        next_x = current_x - velocity * math.cos(current_yaw) * dt
        next_y = current_y + velocity * math.sin(current_yaw) * dt
        next_yaw = current_yaw + velocity * math.sin(steer_angle) * dt / car_l

        return next_x, next_y, next_yaw

    def MPC_forward(self, cmd_list):
        start_x = 65
        start_y = 48
        start_yaw = 0

        predict_x = []
        predict_y = []

        current_x = start_x
        current_y = start_y
        current_yaw = start_yaw

        for i in range(int(len(cmd_list) / 2)):
            cmd_v, cmd_s = cmd_list[2 * i], cmd_list[2 * i + 1]
            next_x, next_y, next_yaw = self.vehicle_kinematics_model(current_x, current_y, current_yaw, cmd_v, cmd_s)
            predict_x.append(next_x)
            predict_y.append(next_y)
            current_x = next_x
            current_y = next_y
            current_yaw = next_yaw

        return predict_x, predict_y

    def debug_forward(self, cmd_list):
        """
        Show the reference line and route points
        :param cmd_list:
        :return:
        """
        start_x = 65
        start_y = 48
        start_yaw = 0

        predict_x = []
        predict_y = []

        current_x = start_x
        current_y = start_y
        current_yaw = start_yaw

        for i in range(int(len(cmd_list) / 2)):
            cmd_v, cmd_s = cmd_list[2 * i], cmd_list[2 * i + 1]
            next_x, next_y, next_yaw = self.vehicle_kinematics_model(current_x, current_y, current_yaw, cmd_v, cmd_s)
            predict_x.append(next_x)
            predict_y.append(next_y)
            current_x = next_x
            current_y = next_y
            current_yaw = next_yaw

        img = np.zeros((100, 100), np.uint8)
        img.fill(0)

        for x, y in zip(predict_x, predict_y):
            cv2.circle(img, (int(y), int(x)), 2, (255, 255, 255), thickness=-1)
        for ref_p in self.ref_line:
            img[ref_p[0]][ref_p[1]] = 180

        dst = cv2.resize(img, (500, 500))
        cv2.imshow("pre", dst)

    def MPC_optimize_func(self, cmd_list):
        # cmd_list = [v0, s0, v1, s1, ......]
        pre_x, pre_y = self.MPC_forward(cmd_list)

        ref_error = 0
        sum_s = 0
        sum_centripetal_acc = 0
        sum_acc = 0
        sum_df = 0

        for i in range(int(len(cmd_list) / 2) - 1):
            sum_acc += abs(cmd_list[2 * i] - cmd_list[2 * i + 2])
            sum_df += abs(cmd_list[2 * i + 1] - cmd_list[2 * i + 3])

        last_x, last_y = 65, 48
        for i, [x, y] in enumerate(zip(pre_x, pre_y)):
            distance, _ = self.ref_kd_tree.query([x, y], k=1)
            ref_error += distance
            sum_s += math.sqrt((x - last_x) ** 2 + (y - last_y) ** 2)
            sum_centripetal_acc += cmd_list[2 * i] * math.tan(cmd_list[2 * i + 1])
            last_x, last_y = x, y

        return 1 * ref_error - 1 * sum_s + 0.01 * sum_centripetal_acc + 300 * sum_acc + 10 * sum_df

    def MPC_solve(self):
        # scipy.optimize.minimize can only optimize one-dimensional vectors
        init_cmd = [2.0, 0.0]
        cmd_bound = [(0.5, 5), (-math.pi / 4, math.pi / 4)]
        cmd_0 = np.array([init_cmd[i % 2] for i in range(2 * self.forward_step)])
        bound = [cmd_bound[i % 2] for i in range(2 * self.forward_step)]

        test = optimize.minimize(self.MPC_optimize_func, cmd_0,
                                 bounds=bound,
                                 method='L-BFGS-B',
                                 options={'maxiter': 6})
        if self.debug:
            self.debug_forward(test.x)
        self.velocity_controller(test.x[0])
        self.action[0] = test.x[1] / (math.pi / 4)

    def control_task(self, controller):
        if controller == "PID":
            self.velocity_controller(0.8)
            self.direction_controller()
        elif controller == "MPC":
            self.MPC_solve()
