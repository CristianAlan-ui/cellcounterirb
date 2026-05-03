import numpy as np
import cv2
def HarrisCorner(img):
    img_float = np.float32(img)
    harris = cv2.cornerHarris(img_float, blockSize=2, ksize=3, k=0.04)
    harris = cv2.dilate(harris, None)
    return harris
def HarrisVis(img):
    img_float = np.float32(img)
    harris = cv2.cornerHarris(img_float, blockSize=2, ksize=3, k=0.04)
    harris = cv2.dilate(harris, None)
    harris_vis = img.copy()
    harris_vis[harris > 0.01 * harris.max()] = [0, 0, 255]
    return harris_vis