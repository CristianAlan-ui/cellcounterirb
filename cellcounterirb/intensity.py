import numpy as np

def Logarithmic(img, c):
    img = img.astype(np.float32)
    log_img = np.log(c+img) 
    log_img = (log_img - log_img.min())/(log_img.max()-log_img.min())
    log_img = np.round(log_img*255).astype(np.uint8)
    return log_img

def Contrast(img, m, E):
    img = img.astype(np.float32)
    s = 1 / (1+(m/img))**E
    s = (s - s.min())/(s.max()-s.min())
    s = np.round(s*255).astype(np.uint8)
    return s
