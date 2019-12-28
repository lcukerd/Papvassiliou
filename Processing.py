from statistics import median, mean, variance
from math import floor, ceil, log
from scipy.stats import norm
import math
import numpy as np

from ImageProcessing import *

def applyViterbi(dSPR, strips, pp, width, CCheight, w):
    regions, m0, m1 = getRegions(dSPR, CCheight);
    probabilities = [];
    e0, e1 = getEmission(pp, regions, width, CCheight, w);
    nRegions = [];
    for i in range(len(regions)):
        regionsStrip = regions[i];
        nRegionsStrip = [];
        probabilities.append((0.5 * e1, 0.5 * e0));
        for region in regionsStrip:
            if region[1] < (CCheight / 5):
                continue;
            a00 = math.exp(-region[1]/m0);
            a11 = math.exp(-region[1]/m1);
            a01 = 1 - a00;
            a10 = 1 - a11;
            curr_Text = max (probabilities[-1][0]*a11*e1, probabilities[-1][1]*a01*e1);
            curr_Gap = max (probabilities[-1][0]*a10*e0, probabilities[-1][1]*a00*e0);
            probabilities.append((curr_Text, curr_Gap));

            p = probabilities[-1];
            if p[0] > p[1]:
                nRegionsStrip.append([region[0], region[1], 1]);
            else:
                nRegionsStrip.append([region[0], region[1], 0]);
        nRegions.append(nRegionsStrip);

    return nRegions;

def getEmission(pp, regions, width, CCheight, w):
    e0 = [];
    e1 = [];
    for i in range(len(regions)):
        regionsStrip = regions[i];
        for region in regionsStrip:
            if region[1] < (CCheight / 5):
                continue;
            total = (width if (i+1)*width <= w else (w - i*width));
            value = np.sum(pp[i][region[0]:region[0] + region[1]]);
            if (value == 0):
                continue;
            if (region[2] == 0):
                # starting of Gap
                e0.append(log (value / total));
            else:
                # starting of Text
                e1.append(log (value / total));
    e0 = norm(mean (e0), variance(e0)).pdf(0);
    e1 = norm(mean (e1), variance(e1)).pdf(0);
    return e0, e1;

def getdiffSPR(SPR, image, height):
    (h, w) = np.shape(image);
    dSPR = [];
    for j in range(h):
        total  = 0;
        for k in range(1, height):
            if (j + k < h and j - k >= 0):
                total += k * (SPR[j + k] - SPR[j - k])
        dSPR.append(total);
    dSPR = ((1 / (height * (height + 1))) * np.array(dSPR)).tolist();
    return dSPR;

def getdelta(image, width, strips, pp):
    (h, w) = np.shape(image);
    th = getThreshold(pp, width, strips, image);
    delta = [];
    for i in range(strips):
        total = (width if (i+1)*width <= w else (w - i*width)) * h;
        d = np.sum(pp[i]) / total
        if d > th:
            delta.append(1);
        else:
            delta.append(0);
    return delta

def getThreshold(pp, width, strips, image):
    d = [];
    (h, w) = np.shape(image);
    for i in range(strips):
        total = (width if (i+1)*width <= w else (w - i*width)) * h;
        d.append(np.sum(pp[i]) / total);
    return median(d) / 2

def getWeights(m):
    denom = 0;
    w = [];
    for i in range(-m,m+1):
        denom += math.exp(-3 * abs(i)/(m+1));
    for i in range(-m,m+1):
        w.append(math.exp(-3 * abs(i)/(m+1)) / denom);
    return w;
