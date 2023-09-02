import cv2
from gym import wrappers
from time import time # just to have timestamps in the files
import pygame
from PIL import ImageGrab


class VideoRecorder:
    def __init__(self, path):
        self.path = path
        self.fps = 50
        self.size = (600, 400)
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(path, self.fourcc, self.fps, self.size)

    def record_frame(self):
        screen_content = pygame.display.get_surface()
        print(screen_content)
        image = ImageGrab.grab((0, 0, self.size[0], self.size[1])).convert("RGB")
        print(image)

        self.video_writer.write(image)

    def stop(self):
        self.video_writer.release()
