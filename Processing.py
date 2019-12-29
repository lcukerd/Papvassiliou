from statistics import median, mean, variance
from math import floor, ceil, log
from scipy.stats import norm
import math
import numpy as np

try:
    from ImageProcessing import *
except ModuleNotFoundError:
    from Papvassiliou.ImageProcessing import *

def getEmission(pp, regions, width, CCheight, w):
    e0 = [];
    e1 = [];
    for i in range(len(regions)):
        regionsStrip = regions[i];
        for region in regionsStrip:
            if region[1] < (CCheight / 5):
                continue;
            total = (width if (i+1)*width <= w else (w - i*width)) * region[1];
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

def dist(region, regionN):
    x0 = region[0] + (region[1] / 2)
    x1 = regionN[0] + (regionN[1] / 2)
    return abs(x0-x1);

def minimize(x, SPR, regionsStrip, indexl, indexr):
    t = regionsStrip[indexl] if indexl > 0 else 0 ;
    b = regionsStrip[indexr] if indexr < len(regionsStrip) else len(SPR);
    mini = 99999999;
    pos = -1;

    for i in range(t + 1, b):
        val = (abs(i - x) + 1) * (SPR[i] + 1);
        if val < mini:
            mini = val;
            pos = i;
    if (pos == -1):
        print (t,b)
    return pos;
