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
    draw = False
    for i in range(len(nRegions)):
        regionsStrip = nRegions[i];
        for region in regionsStrip:
            if (region[2] == 0 and draw == True):
                draw = False;
                cv.line(imageN, (i*width, region[0] + int (region[1] / 2)), ((i+1)*width if (i+1)*width < w else w,  region[0] + int (region[1] / 2)), (0,0,255), 3, cv.LINE_AA);
            elif region[2] == 1:
                draw = True;
            elif region[2] == 2:
                cv.line(imageN, (i*width, region[0]), ((i+1)*width if (i+1)*width < w else w,  region[0]), (0,0,255), 3, cv.LINE_AA);
    plt.imshow(imageN);
    plt.show()

def showProcessedRegions(image, nRegions, width, w):
    imageN = np.copy(image);
    draw = False
    for i in range(len(nRegions)):
        regionsStrip = nRegions[i];
        for region in regionsStrip:
            cv.line(imageN, (i*width, region), ((i+1)*width if (i+1)*width < w else w,  region), (0,0,255), 3, cv.LINE_AA);
    plt.imshow(imageN);
    plt.show()

def showCC(stats, image):
    image = np.copy(image);
    edgyImg = cv.Canny(image, 50, 200, None, 3)
    edgyColor = cv.cvtColor(edgyImg, cv.COLOR_GRAY2BGR)
    DemoImg = np.zeros_like(edgyColor);

    for stat in stats:
        pt1 = (stat[0]          , stat[1]          )
        pt2 = (stat[0] + stat[2], stat[1]          )
        pt3 = (stat[0] + stat[2], stat[1] + stat[3])
        pt4 = (stat[0]          , stat[1] + stat[3])
        cv.line(image, pt1, pt2, (0,0,255), 1, cv.LINE_AA)
        cv.line(image, pt2, pt3, (0,0,255), 1, cv.LINE_AA)
        cv.line(image, pt3, pt4, (0,0,255), 1, cv.LINE_AA)
        cv.line(image, pt4, pt1, (0,0,255), 1, cv.LINE_AA)
        cv.line(DemoImg, pt1, pt2, (0,0,255), 1, cv.LINE_AA)
        cv.line(DemoImg, pt2, pt3, (0,0,255), 1, cv.LINE_AA)
        cv.line(DemoImg, pt3, pt4, (0,0,255), 1, cv.LINE_AA)
        cv.line(DemoImg, pt4, pt1, (0,0,255), 1, cv.LINE_AA)
    plt.imshow(DemoImg);
    plt.show();
    plt.imshow(image);
    plt.show();
