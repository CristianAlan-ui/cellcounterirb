import numpy as np
import cv2
def HarrisCorner(img, blockSize, ksize, k):
     """
    Detects the Harris corner response of an input image.

    Parameters:
        img (ndarray): Input image (must be in gray scale).
        blockSize (int): Distance or neighborhood used for the corner detection.
        ksize (int): Parameter of sobel operator.
        k (float): Sensitivity.
    Returns:
        harris (ndarray): Harris response map indicating corner strength.
    """
    img_float = np.float32(img)
    harris = cv2.cornerHarris(img_float, blockSize, ksize, k)
    harris = cv2.dilate(harris, None)
    return harris
def HarrisVis(img, blockSize, ksize, k):
    """
    Detects and visualizes Harris corners on the input image.
    Parameters:
        img (ndarray): Input image (must be grayscale).
        blockSize (int): Distance or neighborhood used for the corner detection.
        ksize (int): Parameter of sobel operator.
        k (float): Sensitivity.
    Returns:
        harris_vis (ndarray): Original image with detected corners highlighted in red.
    """
    img_float = np.float32(img, blockSize, ksize, k)
    harris = cv2.cornerHarris(img_float, blockSize, ksize, k)
    harris = cv2.dilate(harris, None)
    harris_vis = img.copy()
    harris_vis[harris > 0.01 * harris.max()] = [0, 0, 255]
    return harris_vis
    
