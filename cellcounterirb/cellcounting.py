import cv2
import numpy as np
from scipy import signal


def CountCells(img, Hx, freq, dist, size, loga, umb, aa):

    img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dx = signal.convolve2d(img, Hx, mode='same')


    Hy = Hx.T
    dy = signal.convolve2d(img, Hy, mode='same')
    img = np.sqrt(dx**2 + dy**2)


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
    img = np.uint8(img_filt)
    
    
    kp = np.ones((size,size))
    kp = kp/sum(kp.ravel())
    img = cv2.filter2D(img,-1,kp)
    
    
    img = img.astype(np.float32) #agregado para trabajar en float
    log_img = np.log(loga+img) 
    log_img = (log_img - log_img.min())/(log_img.max()-log_img.min())
    img = np.round(log_img*255).astype(np.uint8)


    img[img >= umb] = 255
    img[img <= umb] = 0


    img = img-255
    img[0:20,:] = 1
    img[-20:-1,:] = 1
    img[:,0:20] = 1
    img[:,-20:-1] = 1
 

     
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    
    
    umbral, binaria = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binaria, 8)
    
    resultado = img.copy()
    conteo = 0
    
    for i in range(1, num_labels):   # 0 = fondo
        x, y, w, h, area = stats[i]
    
        # Filtrar regiones pequeñas para evitar ruido
        if area > aa:
            conteo += 1
            cv2.rectangle(resultado, (x, y), (x + w, y + h), (0, 0, 255), 2)

    img = img + i
    return conteo, resultado
