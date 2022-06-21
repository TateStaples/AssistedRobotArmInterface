import math
import numpy as np
import cv2

video = cv2.VideoCapture(0)


def border(img, color=(255, 0, 0), thickness=10, length=30, spacing=10):
    h, w = img.shape[:2]
    x = y = 0
    while x < w:
        far = min(w, x + length)
        cv2.rectangle(img, (x, 0), (far, thickness), color, thickness=-1)
        cv2.rectangle(img, (x, h), (far, h-thickness), color, thickness=-1)
        x += length + spacing
    while y < h:
        far = min(h, y + length)
        cv2.rectangle(img, (0, y), (thickness, far), color, thickness=-1)
        cv2.rectangle(img, (w, y), (w-thickness, far), color, thickness=-1)
        y += length + spacing
    # print(img.shape)
    return img


if __name__ == '__main__':
    while True:
        ret, frame = video.read()
        if ret:
            img = border(frame, length=50)
            cv2.imwrite("test.png", img)
            break