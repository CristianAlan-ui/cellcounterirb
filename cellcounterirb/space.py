import numpy as np
import cv2
from scipy import signal
def Average(img, tam):
    """
    Applies an average filter to smooth an image.

    Parameters:
        img (ndarray): Input image.
        tam (int): Size of the square kernel (tam x tam).

    Returns:
        av (ndarray): Smoothed image.
    """
    kp = np.ones((tam,tam))
    kp = kp/sum(kp.ravel())
    av = cv2.filter2D(img,-1,kp)
    return av
def Gaussian(img, size, sigma):
    """
    Applies a Gaussian blur filter to smooth an image.

    Parameters:
        img (ndarray): Input image.
        size (int): Size of the Gaussian kernel (must be odd).
        sigma (float): Standard deviation of the Gaussian distribution.

    Returns:
        img (ndarray): Gaussian image.
    """
    m = size//2
    x,y = np.meshgrid(np.arange(-m,m+1), np.arange(-m, m+1))
    k = (1/(2*3.1416*sigma**2))*np.exp(-(x**2+y**2)/(2*sigma**2))
    k = k/sum(k.ravel())
    img = cv2.filter2D(img, -1,k)
    return img
def Convolve(img, h):
    """
    Applies a 2D convolution between an image and a kernel.

    Parameters:
        img (ndarray): Input image.
        h (ndarray): Convolution kernel.

    Returns:
        conv (ndarray): Result of convolution operation.
    """
    h = h[::-1, ::-1]
    conv = signal.convolve2d(img, h, mode = 'same')
    return conv

