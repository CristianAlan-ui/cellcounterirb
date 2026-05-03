import numpy as np
def GrayScale(img, mode):
    """
    Converts an input image into gray scale
    Parameters:
        mode (int): Image format mode.
            1: RGB input image.
            2: BGR input image.
    Returns:
        gray (ndarray): gray scale version of the image
    """
    if(mode == 0):
        r = img[:,:,0]
        g = img[:,:,1]
        b = img[:,:,2]
    elif (mode == 1):
        b = img[:,:,0]
        g = img[:,:,1]
        r = img[:,:,2]
    else:
        print('El modo tiene que ser 0 para rgb y 1 para bgr se hara en modo rgb en esta ocasion')
        r = img[:,:,0]
        g = img[:,:,1]
        b = img[:,:,2]
    gray = 0.2989*r+0.5870*g+0.1140*b
    return gray.astype(np.uint8)

