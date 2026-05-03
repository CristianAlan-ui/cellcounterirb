import numpy as np
import cv2
def LowPass(img, freq):
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