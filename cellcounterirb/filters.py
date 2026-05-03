import numpy as np
import cv2
def LowPass(img, freq):
    def HoughCircle(img):
    """
    Applies a low-pass filter in the frequency domain and converts the result
    back to the spatial domain.
    
    Parameters:
        img (ndarray): Input image for the low-pass filter (must be grayscale).
        freq(int): cuttoff frequency of the filter
    Returns:
        img_filt (ndarray): Filtered image in the spatial domain.
    """
    rows, cols = img.shape
    crow, ccol = rows//2,cols//2
    mask = np.zeros((rows,cols))
    mask = cv2.circle(mask, (ccol,crow), freq,1,-1)
    fft = np.fft.fft2(img)
    fftshift = np.fft.fftshift(fft)
    fft_filt = fftshift*mask
    reg = np.fft.ifftshift(fft_filt)
    reg = np.fft.ifft2(reg)
    img_filt = np.abs(reg)
    img_filt = cv2.normalize(img_filt, None, 0, 255, cv2.NORM_MINMAX)
    img_filt = np.uint8(img_filt)
    return img_filt
def HighPass(img, freq):
    """
    Applies a high-pass filter in the frequency domain and converts the result
    back to the spatial domain.
    
    Parameters:
        img (ndarray): Input image for the high-pass filter (must be grayscale).
        freq(int): cuttoff frequency of the filter
    Returns:
        img_filt (ndarray): Filtered image in the spatial domain.
    """
    rows, cols = img.shape
    crow, ccol = rows//2,cols//2
    mask = np.ones((rows, cols))
    mask = cv2.circle(mask, (ccol, crow), freq,0,-1 )
    fft = np.fft.fft2(img)
    fftshift = np.fft.fftshift(fft)
    fft_filt = fftshift*mask
    reg = np.fft.ifftshift(fft_filt)
    reg = np.fft.ifft2(reg)
    img_filt = np.abs(reg)
    img_filt = cv2.normalize(img_filt, None, 0, 255, cv2.NORM_MINMAX)
    img_filt = np.uint8(img_filt)
    return img_filt
def BandPass(img, freq, dist):
    """
    Applies a band-pass filter in the frequency domain and converts the result
    back to the spatial domain.
    
    Parameters:
        img (ndarray): Input image for the band-pass filter (must be grayscale).
        freq(int): cuttoff frequency of the filter
        dist(int): Sets the band frequency in the filter (starting from the freq parameter)
    Returns:
        img_filt (ndarray): Filtered image in the spatial domain.
    """
    rows, cols = img.shape
    crow, ccol = rows//2,cols//2
    mask = np.zeros((rows, cols))
    mask = cv2.circle(mask, (ccol, crow), freq,1,dist)
    fft = np.fft.fft2(img)
    fftshift = np.fft.fftshift(fft)
    fft_filt = fftshift*mask
    reg = np.fft.ifftshift(fft_filt)
    reg = np.fft.ifft2(reg)
    img_filt = np.abs(reg)
    img_filt = cv2.normalize(img_filt, None, 0, 255, cv2.NORM_MINMAX)
    img_filt = np.uint8(img_filt)
    return img_filt
def BandStop(img, freq, dist):
    """
    Applies a band-stop filter in the frequency domain and converts the result
    back to the spatial domain.
    
    Parameters:
        img (ndarray): Input image for the band-stop filter (must be grayscale).
        freq(int): cuttoff frequency of the filter
        dist(int): Sets the band frequency in the filter (starting from the freq parameter)
    Returns:
        img_filt (ndarray): Filtered image in the spatial domain.
    """
    rows, cols = img.shape
    crow, ccol = rows//2,cols//2
    mask = np.ones((rows, cols))
    mask = cv2.circle(mask, (ccol, crow), freq,0,dist)
    fft = np.fft.fft2(img)
    fftshift = np.fft.fftshift(fft)
    fft_filt = fftshift*mask
    reg = np.fft.ifftshift(fft_filt)
    reg = np.fft.ifft2(reg)
    img_filt = np.abs(reg)
    img_filt = cv2.normalize(img_filt, None, 0, 255, cv2.NORM_MINMAX)
    img_filt = np.uint8(img_filt)
    return img_filt
