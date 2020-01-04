from statistics import median, mean, variance
from math import floor, ceil, log
from scipy.stats import norm
import numpy as np
import cv2 as cv

try:
    from Processing import *
except ModuleNotFoundError:
    from Papvassiliou.Processing import *

def applyViterbi(dSPR, strips, pp, width, CCheight, w):
    regions, m0, m1 = getRegions(dSPR, CCheight);
    e0, e1 = getEmission(pp, regions, width, CCheight, w);
    nRegions = [];
    for i in range(len(regions)):
        regionsStrip = regions[i];
        nRegionsStrip = [];
        probabilities = [];
        probabilities.append((0.5 * e1, 0.5 * e0));
        for region in regionsStrip:
            a00 = math.exp(-region[1]/m0);
            a11 = math.exp(-region[1]/m1);
            a01 = 1 - a00;
            a10 = 1 - a11;
            if region[2] == 1:
                curr_Text = max (probabilities[-1][0]*a11*e1, probabilities[-1][1]*a01*e1);
                curr_Gap = max (probabilities[-1][0]*a10*e1, probabilities[-1][1]*a00*e1);
            else:
                curr_Text = max (probabilities[-1][0]*a11*e0, probabilities[-1][1]*a01*e0);
                curr_Gap = max (probabilities[-1][0]*a10*e0, probabilities[-1][1]*a00*e0);
            probabilities.append((curr_Text, curr_Gap));

            p = probabilities[-1];
            if p[0] > p[1]:
                nRegionsStrip.append([region[0], region[1], 1]);
            else:
                nRegionsStrip.append([region[0], region[1], 0]);
        nRegions.append(nRegionsStrip);

    return nRegions;

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

def getCC(image):
    edgyImg = cv.Canny(image, 100, 200, None, 3)
    edgyColor = cv.cvtColor(edgyImg, cv.COLOR_GRAY2BGR)
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(edgyImg);

    return labels, stats, centroids;


def getCCHeight(image):
    _, stats, _ = getCC(image)

    avg_height = 0;
    for stat in stats:
        avg_height += stat[cv.CC_STAT_HEIGHT]
    avg_height /= len(stats)
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

def performMultiAssociation(regionsStrip, j, regionsStripN, index, SPR):
    x0 = regionsStrip[j - 1];
    y0 = minimize(x0, SPR, regionsStripN, index - 1, index);

    regionsStripN[index] = y0;

    x1 = regionsStrip[j];
    y1 = minimize(x1, SPR, regionsStripN, index, index + 1);

    backup = regionsStripN[index + 1:];
    regionsStripN[index + 1:] = [y1];
    regionsStripN[index + 2:] = backup;

    return regionsStripN;

def connectSeparators(pRegions, delta, SPR, h):
    associateN = np.ones((h)) * -1;
    for i in range(len (pRegions)):
        regionsStrip = pRegions[i];
        index = 0;

        if delta[i] == 0:
            continue;
        if i + 1 < len (pRegions):
            regionsStripN = pRegions[i + 1];
        else:
            regionsStripN = -1;

        associateT = np.copy(associateN);
        associateN = np.ones((h)) * -1;
        for j in range(len(regionsStrip)):
            region = regionsStrip[j];
            if regionsStripN == -1:
                continue;

            index = findNextSeparator(regionsStrip[j], regionsStripN);

            if associateN[index] == 1:
                regionsStripN = performMultiAssociation(regionsStrip, j, regionsStripN, index, SPR[i + 1]);
                associateN[index + 1] = 1;
                index += 1;
            else:
                associateN[index] = 1;
    generateAssociations(pRegions, delta, SPR)
