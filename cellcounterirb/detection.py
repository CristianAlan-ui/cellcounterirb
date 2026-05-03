import cv2
import numpy as np
def HoughLine(img, k, sigma, L, H):
    """
    Detects lines in an image using the Hough Transform after edge detection.
    Parameters:
        img (ndarray): Input image for line detection.
        k (int): Kernel size for Gaussian blur (must be an odd number).
        sigma (float): Standard deviation for Gaussian blur.
        L (int): Lower threshold for Canny edge detection.
        H (int): Upper threshold for Canny edge detection.

    Returns:
        line_img (ndarray): Image with detected lines drawn in green.
    """
    blur = cv2.GaussianBlur(img, (k, k), sigma)
    edges_canny = cv2.Canny(blur,L,H) 
    line_img = img.copy()
    linesP = cv2.HoughLinesP(
        edges_canny,
        rho=1,
        theta=np.pi/180,
        threshold=80,
        minLineLength=40,
        maxLineGap=10
    )

    if linesP is not None:
        for l in linesP:
            x1, y1, x2, y2 = l[0]
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return line_img

def HoughCircle(img):
     """
    Detects circles in an image using the Hough Circle Transform.

    Parameters:
        img (ndarray): Input image for circle detection (input in grey scale).

    Returns:
        circles_vis (ndarray): Image with detected circles and centers drawn.
    """
    circles_vis = img.copy()

    circles = cv2.HoughCircles(
        img,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=40,
        param1=100,
        param2=20,
        minRadius=20,
        maxRadius=90
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        for (x, y, r) in circles:
            cv2.circle(circles_vis, (x, y), r, (0, 255, 0), 2)
            cv2.circle(circles_vis, (x, y), 2, (0, 0, 255), 3)
    return circles_vis
