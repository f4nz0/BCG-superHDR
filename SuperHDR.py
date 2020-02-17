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
        self.displacement = (0, 0)

    def __str__(self):
        out = "Appropriate: " + str(self.appr_count) + "\n"
        out += "Blacked out: " + str(self.blk_count) + "\n"
        out += "Saturated: " + str(self.sat_count) + "\n"
        return out


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
    bi = sdrimage.binarized
    classification = sdrimage.classification
    for x in range(0, tri.shape[0]):
        for y in range(0, tri.shape[1]):
            if classification == "ref":
                if tri[x][y] == 1:
                    bi[x][y] == 1
                else:
                    bi[x][y] == 0
            elif classification == "blackedout":
                if tri[x][y] == 0:
                    bi[x][y] == 1
                else:
                    bi[x][y] == 0
            elif classification == "saturated":
                if tri[x][y] == 2:
                    bi[x][y] == 1
                else:
                    bi[x][y] == 0


def cvt_binarized2grayscale(bi):
    grayimg = np.zeros(bi.shape, dtype=np.uint8)
    for x in range(0, bi.shape[0]):
        for y in range(0, bi.shape[1]):
            if bi[x][y] == 0:
                grayimg[x][y] = 70
            else:
                print("HELLO")
                grayimg[x][y] = 180
    return grayimg


def sort_into_chain(ref, sdrimage):
    if sdrimage.blk_count > ref.blk_count:
        sdrimage.classification = "blackedout"
        if ref.darker is None:
            ref.darker = sdrimage
            sdrimage.brighter = ref
        else:
            darker = ref.darker
            if darker.blk_count >= sdrimage.blk_count:
                sdrimage.brighter = ref
                sdrimage.darker = darker
                darker.brighter = sdrimage
                ref.darker = sdrimage
            else:
                sort_into_chain(darker, sdrimage)
    elif sdrimage.sat_count > ref.sat_count:
        sdrimage.classification = "saturated"
        if ref.brighter is None:
            ref.brighter = sdrimage
            sdrimage.darker = ref
        else:
            brighter = ref.brighter
            if brighter.sat_count >= sdrimage.sat_count:
                sdrimage.darker = ref
                sdrimage.brighter = brighter
                brighter.darker = sdrimage
                ref.brighter = sdrimage
            else:
                sort_into_chain(brighter, sdrimage)


def convert_sdr2hdr(sdr_series):
    # CLASSIFIYING INTO EITHER BLACKET OUT, REFERENCE OR SATURATED
    ref = ex_classification(sdr_series)

    # for i in range(0, len(sdr_series)):
    #     cv2.imshow("img" + str(i), cvt_trinarized2grayscale(sdr_series[i].trinarized))
    #     if ref == sdr_series[i]:
    #         print("FOUND", i)
    #     print(i, sdr_series[i].appr_count)





def ex_classification(sdr_series):
    ref = sdr_series[1]
    for sdrimage in sdr_series:
        trinarize(sdrimage)
        if sdrimage.appr_count > ref.appr_count:
            ref = sdrimage
    ref.classification = "ref"
    for sdrimage in sdr_series:
        sort_into_chain(ref, sdrimage)
        binarize(sdrimage)
    cv2.imshow("reference tri", cvt_trinarized2grayscale(ref.trinarized))
    cv2.imshow("reference bin", cvt_binarized2grayscale(ref.binarized))
    return ref


    # print("SORTED")
    # out = str(ref)
    # darker = ref.darker
    # while darker is not None:
    #     out = str(darker) + out
    #     darker = darker.darker
    # brighter = ref.brighter
    # while brighter is not None:
    #     out += str(brighter)
    #     brighter = brighter.brighter
    # print(out)
