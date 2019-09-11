import numpy as np
import cv2

def add_background(img, background):
    img = np.array(img)
    background = cv2.resize(background, tuple(reversed(img.shape[:2])))

    background_pixels = np.where(img == 0)
    img[background_pixels] = background[background_pixels]

    return img
