import cv2
import numpy as np
from numba import jit

@jit(nopython=True)
def find_first(item, vec):
    """return the index of the first occurence of item in vec"""
    for i in range(len(vec)):
        if vec[i] >= item:
            return i
    return -1

@jit(nopython=True)
def find_end(item, vec):
    """return the index of the first occurence of item in vec"""
    for i in range(len(vec)-1, 0, -1):
        if item >= vec[i]:
            return i
    return -1

def imageStetch(img: np.array, percent: float):
    if img.dtype == np.uint8:
        length = 1 << 8
    elif img.dtype == np.uint16:
        length = 1 << 16
    else:
        raise NotImplementedError
    hist = cv2.calcHist([img], [0], None, [length], [0, length])
    hist_p = np.cumsum(hist) / float(img.size)
    min_value = find_first(percent, hist_p)
    max_value = find_first(1 - percent, hist_p)
    new_img = 255.0 * (img - min_value) / (max_value - min_value)
    np.clip(new_img, 0, 255, new_img)

    return new_img.astype(np.uint8)

