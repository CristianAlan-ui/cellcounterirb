# CellCounterIRB
Librería para el procesamiento de imágenes y conteo automático de células mediante técnicas de filtrado y análisis de bordes.

## Requisitos:
* Tener instalado python
* Tener instalado 'pip'
* Tener instalado git y tenerlo agregado al path 

## Proceso de instalación:
* Abrir terminal
```
pip install git+https://github.com/CristianAlan-ui/cellcounterirb.git
```

## Uso
### Funcion principal
*CountCells
### Catalogo completo de funciones:

*HarrisCorner
*HarrisVis
*Binary
*Otsu
*Average
*Gaussian
*Convolve
*Gradient
*Sobel
*Prewitt
*Laplacian
*LaplacianD
*Canny
*GrayScale
*Logarithmic
*Contrast
*HoughLine
*HoughCircle
*LowPass
*HighPass
*BandPass
*BandStop
*CountCells
Nota: Consultar docstrings para informacion de los argumentos
### Ejemplo de uso de CountCells
```
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cellcounterirb as cl

img = cv2.imread('img.jpg')
Hx = np.array([[-1, 0, 1],
               [-1, 0, 1],
               [-1, 0, 1]])
count, I = cl.CountCells(img, Hx, 15, 30,1, 0.5,180,150)

plt.figure(1)
plt.clf()
plt.subplot(2,1,1), plt.imshow(I, cmap = 'gray'), plt.title('Imagen con Conteo')
plt.subplot(2,1,2), plt.imshow(img, cmap = 'gray'), plt.title('Imagen original')
print(count)
```
## Notas

* Es importante que la versión de Python utilizada para instalar la librería coincida con la del entorno donde se ejecutará.
* Si 'git' o 'pip' no están correctamente configurados, la instalación puede fallar.
