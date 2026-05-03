import cv2
def Binary(img, umbral):
    """
    Applies a global binary threshold to an image.
    Parameters:
        img (ndarray): Input grayscale image.
        umbral (int): Threshold value. Pixels greater than this value are set to 255,
            otherwise they are set to 0.

    Returns:
        binary (ndarray): Binary image resulting.
    """
    th_global = cv2.threshold(img, umbral, 255, cv2.THRESH_BINARY)
    return th_global[1]

def Otsu(img):
    """
    Applies Otsu's automatic thresholding method for image binarization.

    Parameters:
        img (ndarray): Input grayscale image.

    Returns:
        binary (ndarray): Binary image obtained using Otsu thresholding.
    """
    th_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th_otsu[1]
