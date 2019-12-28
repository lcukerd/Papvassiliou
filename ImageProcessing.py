from statistics import median, mean, variance
from math import floor, ceil, log
from scipy.stats import norm
import numpy as np
import cv2 as cv

def getRegions(dSPR, CCheight):
    regions = [];
    m0 = [];
    m1 = [];
    (strips, h) = np.shape(dSPR)
    for i in range(strips):
        dSPRStrip = dSPR[i];
        regionsS = [];
        minima = -1;
        y = -1;
        for j in range(1, h):
            if dSPRStrip[j] > dSPRStrip[j-1] and minima == 0:
                # Word ends and gap starts
                minima = 1;
                if y != -1:
                    regionsS.append([y, j - y, 1]);
                    if (j - y - 1) > (CCheight / 5):
                        m1.append(j - y - 1);
                y = j;
            elif dSPRStrip[j] < dSPRStrip[j-1]:
                if minima == 1:
                    # Gap ends Word starts
                    minima = 0;
                    if y != -1:
                        regionsS.append([y, j - y, 0]);
                        if (j - y - 1) > (CCheight / 5):
                            m0.append(j - y - 1);
                    y = j;
                elif minima == -1:
                    # First Word starts
                    minima = 0;
                    regionsS.append([0, j - 1, 0]);
                    if (j - 1) > (CCheight / 5):
                        m0.append(j - 1);
                    y = j -1

        regions.append(regionsS);

    return regions, mean(m0), mean(m1);

def getCCHeight(image):
    edgyImg = cv.Canny(image, 50, 200, None, 3)
    edgyColor = cv.cvtColor(edgyImg, cv.COLOR_GRAY2BGR)
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(edgyImg);

    avg_height = 0;
    for stat in stats:
        avg_height += stat[cv.CC_STAT_HEIGHT]
    avg_height /= num_labels
    if floor(avg_height) % 2 == 1:
        return floor(avg_height);
    else:
        return ceil(avg_height);

def getSPR(image, M, zone, delta, wt, pp, strips):
    (h, w) = np.shape(image);
    SPR = h * [0]
    for j in range(-M, M + 1):
        if (zone + j >= 0 and zone + j < strips):
            SPR += delta[zone + j] * wt[j + M] * np.array(pp[zone + j]);
    return SPR.tolist();

def projectionProfile(image, width, i):
    pp = [];
    (h, w) = np.shape(image);
    for y in range(0,h):
        sum = 0;
        for x in range(i*width, (i+1)*width if (i+1)*width < w else w):
            if (image[y,x] == 0):
                sum += 1;
        pp.append(sum);
    return pp;
