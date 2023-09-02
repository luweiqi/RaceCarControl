import math
import cv2
import numpy as np
from sklearn import linear_model
from scipy import optimize


def distance(p1, p2):
    # if your python version >= 3.8, you can use math.dist instead
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def func(x, a, b):
    return a * x + b


class BEV_Vision:
    def __init__(self, debug=True):
        self.src_img = None
        self.img_HSV = None
        self.img_gray = None
        self.img_list = [None, None]
        self.lane_mask = None
        self.ref_line = None
        self.anchor = [70, 48]

        self.color_dist = {'Lane': {'Lower': np.array([0, 0, 90]), 'Upper': np.array([10, 10, 120])}}
        self.min_area = 20
        self.lane_area_min = 600
        self.lane_area_max = 5000
        self.debug = debug
        self.neighborhood = [[1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0]]

        self.sample_step = 25
        self.sample_points = [[x, y] for x in range(10, 64, self.sample_step) for y in range(10, 95, self.sample_step)]
        self.sample_points_r = [[p[0] - self.anchor[0], p[1] - self.anchor[1]] for p in self.sample_points]

        self.velocity = [0, 0, 0]

    def get_raw_image(self, img):
        self.src_img = img[0:65, :]
        self.img_HSV = cv2.cvtColor(self.src_img, cv2.COLOR_BGR2HSV)
        self.img_gray = cv2.cvtColor(self.src_img, cv2.COLOR_RGB2GRAY)
        self.img_list.pop(0)
        self.img_list.append(self.img_gray)
        if self.img_list[0] is None:
            self.img_list[0] = self.img_gray

        # show raw image
        if self.debug:
            cv2.imshow("raw_img", self.src_img)

    def lane_detector(self):
        if self.img_HSV is None:
            print("Error! Empty HSV Image.")
            return

        lane_mask = cv2.inRange(self.img_HSV, self.color_dist['Lane']['Lower'], self.color_dist['Lane']['Upper'])

        contours, hierarch = cv2.findContours(lane_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area < self.min_area:
                cv2.drawContours(lane_mask, [contours[i]], -1, (0, 0, 0), thickness=-1)
            elif self.lane_area_min <= area <= self.lane_area_max:
                cv2.drawContours(lane_mask, [contours[i]], -1, (255, 255, 255), thickness=-1)

        self.lane_mask = lane_mask

        if self.debug:
            cv2.imshow("lane mask", lane_mask)

    def reference_line_extractor(self):
        if self.lane_mask is None:
            print("Error! Empty Lane Mask.")
            return

        # get center line
        thin = cv2.ximgproc.thinning(self.lane_mask)
        thin[0, :] = 0
        thin[64, :] = 0
        thin[:, 0] = 0
        thin[:, 95] = 0

        # sort the points
        self.ref_line = []
        close_list = []

        p_x, p_y = np.where(np.array(thin) == 255)
        if len(p_x) == 0:
            return

        points = [[p_x[i], p_y[i]] for i in range(len(p_x))]
        dist = list(map(distance, points, [self.anchor for i in range(len(points))]))
        start_point = points[dist.index(min(dist))]
        self.ref_line.append(start_point)
        close_list.append(start_point)
        current_pt = start_point

        while True:
            ct = 0
            for nb in self.neighborhood:
                nb_point = [current_pt[0] + nb[0], current_pt[1] + nb[1]]
                if points.count(nb_point) != 0 and close_list.count(nb_point) == 0:
                    self.ref_line.append(nb_point)
                    close_list.append(current_pt)
                    current_pt = nb_point
                    ct += 1
                    continue
            if ct == 0:
                break

        if self.debug:
            print("******** start point ********")
            print(start_point)
            print("                             ")
            cv2.imshow("ref line", thin)
            print("******** reference line ********")
            print(self.ref_line)
            print("                                ")

    def optical_flow(self):
        flow = cv2.calcOpticalFlowFarneback(prev=self.img_list[0], next=self.img_list[1], flow=None, pyr_scale=0.5,
                                            levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

        fx = np.array([np.array([self.sample_points_r[i][0],
                                 self.sample_points_r[i][1]]).T for i in range(len(self.sample_points))]).T
        fy = np.array([np.array([flow[self.sample_points[i][0]][self.sample_points[i][1]][0],
                                 flow[self.sample_points[i][0]][self.sample_points[i][1]][1]])
                       for i in range(len(self.sample_points))]).T

        def optimize_func(x):
            # x = [vx, vy, w]
            A = np.array([[0, -x[2]], [x[2], 0]])
            B = np.array([x[0], x[1]]).T

            np.dot(A, fx)
            np.dot(A, fx) + np.array([B for _ in range(len(self.sample_points))]).T
            s = np.dot(A, fx) + np.array([B for _ in range(len(self.sample_points))]).T - fy

            sum_error = 0
            for x, y in zip(s[0], s[1]):
                sum_error += math.sqrt(x**2 + y**2)

            return sum_error

        test = optimize.minimize(optimize_func, self.velocity, method='BFGS')
        self.velocity = test.x

        if self.debug:
            print("******* velocity vector ********")
            cv2.imshow("gray", self.img_gray)
            print(self.velocity)
            print("                                ")

    def vision_task(self, raw_img):
        self.get_raw_image(raw_img)
        self.lane_detector()
        self.reference_line_extractor()
        self.optical_flow()
        return self.velocity, self.ref_line
