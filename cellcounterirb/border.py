import cv2
from scipy import signal
import numpy as np

def Gradient(img, Hx):
    """
    Computes the gradient of an image using a kernel.
    
    Parameters:
        img (ndarray): Input image.
        Hx (ndarray): Kernel used for horizontal gradient (inverse is the vertical gradient).

    Returns:
        grad (ndarray): Gradient image.
    """
    dx = signal.convolve2d(img, Hx, mode='same')
    Hy = Hx.T
    dy = signal.convolve2d(img, Hy, mode='same')
    grad = np.sqrt(dx**2 + dy**2)
    return grad

def Sobel(img):
     """
    Detects edges using the Sobel operator.

    Parameters:
        img (ndarray): Input image.

    Returns:
        sobel (ndarray): Sobel edge image
    """
    kernel_sx = np.array([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]], dtype=np.float32)

    kernel_sy = np.array([[ 1,  2,  1],
                          [ 0,  0,  0],
                          [-1, -2, -1]], dtype=np.float32)

    edges_sx = signal.convolve2d(img, kernel_sx, mode ='same')
    edges_sy = signal.convolve2d(img, kernel_sy, mode ='same')

    sobel = cv2.magnitude(edges_sx, edges_sy)
    return sobel

def Prewitt(img):
    """
    Detects edges using the Prewitt operator.
    Parameters:
        img (ndarray): Input image.

    Returns:
        prewitt (ndarray): Prewitt edges image.
    """
    kernel_px = np.array([[-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]], dtype=np.float32)

    kernel_py = np.array([[ 1,  1,  1],
                          [ 0,  0,  0],
                          [-1, -1, -1]], dtype=np.float32)

    edges_px = signal.convolve2d(img, kernel_px, mode ='same')
    edges_py = signal.convolve2d(img, kernel_py, mode ='same')
    prewitt = cv2.magnitude(edges_px, edges_py)
    return prewitt

def Laplacian(img):
    """
    Detects edges using the Laplacian operator.
    
    Parameters:
        img (ndarray): Input image.

    Returns:
        edges1 (ndarray): Edge map using Laplacian operator.
    """
    laplace = np.array([[0, 1, 0],
                            [1,-4, 1],
                            [0, 1, 0]])
    edges1 = signal.convolve2d(img, laplace, mode ='same')
    edges1 = np.abs(edges1)
    return edges1

def LaplacianD(img):
    """
    Detects edges using the diagonal Laplacian operator.
    Parameters:
        img (ndarray): Input image.

    Returns:
        edges2 (ndarray): Diagonal edges using diagonal Laplacian operator.
    """
    laplace = np.array([[1, 1, 1],
                            [1,-8, 1],
                            [1, 1, 1]])
    edges2 = signal.convolve2d(img, laplace, mode ='same')
    edges2 = np.abs(edges2)
    return edges2
    
def Canny(img, k, sigma, L, H):
    """
    Detects edges using the Canny edge detection algorithm.

    Parameters:
        img (ndarray): Input image.
        k (int): Kernel size for Gaussian blur (must be odd).
        sigma (float): Standard deviation for Gaussian blur.
        L (int): Lower threshold for edge tracking.
        H (int): Upper threshold for edge tracking.

    Returns:
        edges_canny (ndarray): Binary edge map using canny.
    """
    blur = cv2.GaussianBlur(img, (k, k), sigma)
    edges_canny = cv2.Canny(blur,L,H) 
    return edges_canny
