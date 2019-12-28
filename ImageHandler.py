from matplotlib import pyplot as plt
from pylab import rcParams
import cv2 as cv
import math
import numpy as np

def loadImage(fileName):
    src = cv.imread(cv.samples.findFile(fileName), 0)
    if src is None:
        print ('Error opening image!')
    return src

def showLine(image, dSPR, strips, width):
    image = np.copy(image);
    (h, w) = np.shape(image);
    for i in range(strips):
        minima = -1;
        y = 0;
        for j in range(1, h):
            if dSPR[i][ j] > dSPR[i][j-1] and minima == 0:
                minima = 1;
                y = j
            elif dSPR[i][j] < dSPR[i][j-1]:
                if minima == 1:
                    minima = 0;
                    cv.line(image, (i*width,int ((y + j)/2)), ((i+1)*width if (i+1)*width < w else w, int ((y + j)/2)), (0,0,255), 3, cv.LINE_AA)
                else:
                    minima = 0;
    plt.imshow(image);
    plt.show()

def showRegions(image, nRegions, width, w):
    imageN = np.copy(image);
    for i in range(len(nRegions)):
        regionsStrip = nRegions[i];
        for region in regionsStrip:
            if (region[2] == 0):
                cv.line(imageN, (i*width, region[0]), ((i+1)*width if (i+1)*width < w else w,  region[0]), (0,0,255), 3, cv.LINE_AA)
    plt.imshow(imageN);
    plt.show()
