import numpy as np
import cv2
import time


def good_round(num, to):
    return round(num / to) * to


def create_mask():
    w, h = 640, 480
    img = np.zeros((h, w, 4))
    red = (0, 0, 255, 255)  # bgr
    # circle_radius = 50
    # cv2.circle(img, (w // 2, h // 2), circle_radius, red, -1)
    x1 = 200
    width = 40
    height = 100
    space = 300
    cv2.rectangle(img, (x1, h), (x1 + width, h - height), red, -1)
    cv2.rectangle(img, (x1 + space, h), (x1 + space + width, h - height), red, -1)

    camera_width = 100
    r = 20
    height = h//2
    cv2.rectangle(img, (w//2-camera_width, height+r), (w//2-camera_width, height+r), red, -1)
    cv2.circle(img, (w//2-camera_width))

    cv2.imwrite("resources/mask.png", img)


def create_config(device, size, depth):
    config = rs.config()
    config.enable_device(device)
    if depth:
        config.enable_stream(rs.stream.depth, size[0], size[1], rs.format.z16, 30)
    config.enable_stream(rs.stream.color, size[0], size[1], rs.format.bgr8, 30)
    return config

def add_message(self, msg, t=5):
    self.msg = msg
    self.time = time.time() + t


def average_depth(depth_img):
    w, h = depth_img.shape
    cropped = depth_img[w//4: w//4 * 3, h//4 : h//4 * 3]
    return np.mean(cropped)

cv2.putp
if __name__ == '__main__':
    print(time.time())
    time.sleep(3)
    print(time.time())