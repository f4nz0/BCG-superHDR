# THIS ALGORITHM IS BASED ON OPENCV TUTORIAL FOR HDR IMAGING: https://docs.opencv.org/3.4/d3/db7/tutorial_hdr_imaging.html
from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np
import argparse
import os


def loadExposureSeq():
    images = []
    times = []
    with open(os.path.join("opencvhdr/", 'list.txt')) as f:
        content = f.readlines()
    for line in content:
        tokens = line.split()
        images.append(cv.imread(os.path.join("opencvhdr/", tokens[0])))
        times.append(1 / float(tokens[1]))
    return images, np.asarray(times, dtype=np.float32)


images, times = loadExposureSeq()
calibrate = cv.createCalibrateDebevec()
response = calibrate.process(images, times)
merge_debevec = cv.createMergeDebevec()
hdr = merge_debevec.process(images, times, response)
tonemap = cv.createTonemap(2.2)
ldr = tonemap.process(hdr)
merge_mertens = cv.createMergeMertens()
fusion = merge_mertens.process(images)
cv.imwrite('fusion.png', fusion * 255)
cv.imwrite('ldr.png', ldr * 255)
cv.imwrite('hdr.hdr', hdr)
