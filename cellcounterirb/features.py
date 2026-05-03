import numpy as np
import cv2

def HarrisCorner(img, blockSize, ksize, k):
    """
    Detects the Harris corner response of an input image.

    Parameters:
        img (ndarray): Input image (must be grayscale).
        blockSize (int): Neighborhood size for corner detection.
        ksize (int): Aperture parameter of Sobel operator.
        k (float): Harris detector free parameter.

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
        img (ndarray): Input image (must be BGR).
        blockSize (int): Neighborhood size for corner detection.
        ksize (int): Aperture parameter of Sobel operator.
        k (float): Harris detector free parameter.

    Returns:
        harris_vis (ndarray): Image with detected corners highlighted in red.
    """
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img

    img_float = np.float32(img_gray)

    harris = cv2.cornerHarris(img_float, blockSize, ksize, k)
    harris = cv2.dilate(harris, None)

    harris_vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    harris_vis[harris > 0.01 * harris.max()] = [0, 0, 255]

    return harris_vis
