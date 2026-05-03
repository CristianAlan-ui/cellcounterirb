import numpy as np

def Logarithmic(img, c):
    """
    Applies logarithmic intensity transformation to enhance dark regions of an image.

    Parameters:
        img (ndarray): Input image.
        c (float): Transformation strength.

    Returns:
        log_img (ndarray): Resulting image of the logarithmic transformation
    """
    img = img.astype(np.float32)
    log_img = np.log(c+img) 
    log_img = (log_img - log_img.min())/(log_img.max()-log_img.min())
    log_img = np.round(log_img*255).astype(np.uint8)
    return log_img

def Contrast(img, m, E):
    """
    Applies contrast using a nonlinear transformation
    Parameters:
        img (ndarray): Input image.
        m (float): Midpoint control.
        E (float): Exponent.

    Returns:
        s (ndarray): Returns the transformed image
    """
    img = img.astype(np.float32)
    s = 1 / (1+(m/img))**E
    s = (s - s.min())/(s.max()-s.min())
    s = np.round(s*255).astype(np.uint8)
    return s
