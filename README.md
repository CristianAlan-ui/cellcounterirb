# CellCounterIRB
Library for image processing and automatic cell counting using filtering techniques and edge-based analysis.

## Requirements:
* Python
* pip
* git installed and added to path

## Installation:
* Open the terminal and run:
```
pip install git+https://github.com/CristianAlan-ui/cellcounterirb.git
```

## Usage:

### Main function:
*`CountCells`
### Available secondary functions:

* `HarrisCorner`
* `HarrisVis`
* `Binary`
* `Otsu`
* `Average`
* `Gaussian`
* `Convolve`
* `Gradient`
* `Sobel`
* `Prewitt`
* `Laplacian`
* `LaplacianD`
* `Canny`
* `GrayScale`
* `Logarithmic`
* `Contrast`
* `HoughLine`
* `HoughCircle`
* `LowPass`
* `HighPass`
* `BandPass`
* `BandStop`
Note: each function has a docstring with internal documentation.
### CountCells usage example:
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
plt.subplot(2,1,1), plt.imshow(I, cmap = 'gray'), plt.title('Counted image')
plt.subplot(2,1,2), plt.imshow(img, cmap = 'gray'), plt.title('Original Image')
print(count)
```
count: number of detected cells  
I: image with detected cells highlighted
## Notes
* Make sure the version of the interpreter matches with the python where the environment where the library was installed.
* If 'git' or 'pip' are not correctly configured, the installation of the library can fail.
