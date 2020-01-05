import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from pylab import rcParams

try:
    from ImageHandler import *
    from ImageProcessing import *
    from Processing import *
except ModuleNotFoundError:
    from Papvassiliou.ImageHandler import *
    from Papvassiliou.ImageProcessing import *
    from Papvassiliou.Processing import *

def performPapvassiliouSegmentation(file_name):
    image = loadImage(file_name);
    (h, w) = np.shape(image);
    _, image = cv.threshold(image,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

    width = int (0.05 * w);
    (h, w) = np.shape(image);
    strips = (int (w/width) + (1 if w%width != 0 else 0));
    M = 3;

    wt = getWeights(M);
    pp = [];
    for i in range(strips):
        pp.append(projectionProfile(image, width, i));

    delta = getdelta(image, width, strips, pp);
    SPR = [];
    for i in range(strips):
        SPR.append(getSPR(image, M, i, delta, wt, pp, strips));

    dSPR = [];
    CCheight = int (getCCHeight(image) / 2);
    for i in range(strips):
        dSPR.append(getdiffSPR(SPR[i], image, CCheight));
    nRegions = applyViterbi(dSPR, strips, pp, width, CCheight, w)

    pRegions = [];
    for regionsStrip in nRegions:
        pRegionsStrip = [];
        tempRegion = [];
        gap = False
        for region in regionsStrip:
            if region[2] == 0:
                if gap:
                    tempRegion[1] += region[1]
                else:
                    if tempRegion != []:
                        pRegionsStrip.append(region[0] + int (region[1] / 2));
                    tempRegion = region;
                    gap = True;
            else:
                gap = False;
        pRegions.append (pRegionsStrip);
    textRegions = connectSeparators(pRegions, delta, SPR, h);

    labels = [];
    stats = [];
    centroids = [];
    for i in range(strips):
        imageStrip = image[:,i*width : (i+1)*width if (i+1)*width < w else w];
        if delta[i] == 1:
            labelStrip, statStrip, centroidStrip = getCC(imageStrip);
        else:
            labelStrip, statStrip, centroidStrip = [], [], [];
        labels.append(labelStrip);
        stats.append(statStrip);
        centroids.append(centroidStrip);

    chosenLines = [];
    for i in range(strips):
        chosenLinesStrip = [];
        for j in range(len(stats[i])):
            stat = stats[i][j]
            if (len (stat) == 0):
                break;
            lines = getLineinRange(pRegions[i], stat[1], stat[1] + stat[3]);
            th75 = stat[1] + (0.75 * stat[3]);
            lines = [line for line in lines if line > th75];
            lines = checkDuplicacy(chosenLinesStrip, lines);
            if len(lines) > 1:
                lines = [lines[0]]
    #             lines = extendZone(image, pRegions, i, stats, stat[1], stat[1] + stat[3], lines);
            chosenLinesStrip.extend(lines);
        chosenLines.append(chosenLinesStrip);

    return max([len(strip) for strip in chosenLines]);
