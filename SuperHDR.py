import cv2
import numpy as np

# image sources http://hdr-photographer.com/hdr-photos-to-play-with/

sat_threshold = 220
blk_threshold = 40

class sdrImage:
    def __init__(self, im):
        self.image = im
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.trinarized = np.zeros((im.shape[0], im.shape[1]), dtype=np.uint8)
        self.binarized = np.zeros((im.shape[0], im.shape[1]), dtype=np.uint8)
        self.appr_count = 0
        self.sat_count = 0
        self.blk_count = 0
        self.classification = "none"
        self.brighter = None
        self.darker = None


def trinarize(sdrimage):
    image = sdrimage.image
    for x in range(0, image.shape[0]):
        for y in range(0, image.shape[1]):
            if image[x][y][2] > sat_threshold:
                sdrimage.trinarized[x][y] = 2
                sdrimage.sat_count += 1
            elif image[x][y][2] < blk_threshold:
                sdrimage.trinarized[x][y] = 0
                sdrimage.blk_count += 1
            else:
                sdrimage.trinarized[x][y] = 1
                sdrimage.appr_count += 1


def cvt_trinarized2grayscale(tri):
    grayimg = np.zeros(tri.shape, dtype=np.uint8)
    for x in range(0, tri.shape[0]):
        for y in range(0, tri.shape[1]):
            if tri[x][y] == 0:
                grayimg[x][y] = 0
            elif tri[x][y] == 2:
                grayimg[x][y] = 255
            else:
                grayimg[x][y] = 200
    return grayimg


def binarize(sdrimage):
    tri = sdrimage.trinarized
    bin = sdrimage.binarized
    classification = sdrimage.classification
    for x in range(0, tri.shape[0]):
        for y in range(0, tri.shape[1]):
            if classification == "ref":
                if tri[x][y] == 1:
                    bin[x][y] == 1
                else:
                    bin[x][y] == 0
            elif classification == "blackedout":
                if tri[x][y] == 0:
                    bin[x][y] == 1
                else:
                    bin[x][y] == 0
            else:
                if tri[x][y] == 2:
                    bin[x][y] == 1
                else:
                    bin[x][y] == 0


def cvt_binarized2grayscale(bin):
    grayimg = np.zeros(bin.shape, dtype=np.uint8)
    for x in range(0, bin.shape[0]):
        for y in range(0, bin.shape[1]):
            if bin[x][y] == 0:
                grayimg[x][y] = 0
            else:
                grayimg[x][y] = 1
    return grayimg


def ex_superHDR(sdr_series):
    ref = sdr_series[0]
    for sdr_image in sdr_series:
        trinarize(sdr_image)
        if sdr_image.appr_count > ref.appr_count:
            ref = sdr_image
    return sdr_image