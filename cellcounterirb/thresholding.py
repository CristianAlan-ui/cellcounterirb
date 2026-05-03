import cv2
def Binary(img, umbral):
    th_global = cv2.threshold(img, umbral, 255, cv2.THRESH_BINARY)
    return th_global[1]

def Otsu(img):
    th_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th_otsu[1]
