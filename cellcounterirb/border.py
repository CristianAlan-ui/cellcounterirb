#Border detection
import cv2
from scipy import signal
import numpy as np

def Gradient(img, Hx):
    dx = signal.convolve2d(img, Hx, mode='same')

    Hy = Hx.T
    dy = signal.convolve2d(img, Hy, mode='same')
    grad = np.sqrt(dx**2 + dy**2)
    return grad

def Sobel(img):
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
    laplace = np.array([[0, 1, 0],
                            [1,-4, 1],
                            [0, 1, 0]])
    edges1 = signal.convolve2d(img, laplace, mode ='same')
    edges1 = np.abs(edges1)
    return edges1

def LaplacianD(img):
    laplace = np.array([[1, 1, 1],
                            [1,-8, 1],
                            [1, 1, 1]])
    edges2 = signal.convolve2d(img, laplace, mode ='same')
    edges2 = np.abs(edges2)
    return edges2
    
def Canny(img, k, sigma, L, H):
    blur = cv2.GaussianBlur(img, (k, k), sigma)
    edges_canny = cv2.Canny(blur,L,H) 
    return edges_canny