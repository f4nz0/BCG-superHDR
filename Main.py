import cv2
import numpy as np
from SuperHDR import *

# image sources http://hdr-photographer.com/hdr-photos-to-play-with/


sdr_series = []
sdr_series.append(sdrImage(cv2.imread("Royal_Palace_in_Madrid/img_1335.jpg", cv2.IMREAD_COLOR)))
sdr_series.append(sdrImage(cv2.imread("Royal_Palace_in_Madrid/img_1336.jpg", cv2.IMREAD_COLOR)))
sdr_series.append(sdrImage(cv2.imread("Royal_Palace_in_Madrid/img_1337.jpg", cv2.IMREAD_COLOR)))
sdr_series.append(sdrImage(cv2.imread("Royal_Palace_in_Madrid/img_1338.jpg", cv2.IMREAD_COLOR)))
sdr_series.append(sdrImage(cv2.imread("Royal_Palace_in_Madrid/img_1339.jpg", cv2.IMREAD_COLOR)))
sdr_series.append(sdrImage(cv2.imread("Royal_Palace_in_Madrid/img_1340.jpg", cv2.IMREAD_COLOR)))
sdr_series.append(sdrImage(cv2.imread("Royal_Palace_in_Madrid/img_1341.jpg", cv2.IMREAD_COLOR)))

# cv2.imshow("LOL", img[0].image)


def convert_sdr2hdr(sdr_series):
    ref = ex_superHDR(sdr_series)
    cv2.imshow(str("img"), cvt_trinarized2grayscale(ref.trinarized))


    # highest_appr_index = 0
    # for i in range(0, len(sdr_series)):
    #     trinarize(sdr_series[i])
    #     cv2.imshow(str("img" + str(i)), cvt_trinarized2grayscale(sdr_series[i].trinarized))
    #     if sdr_series[i].appr_count > sdr_series[highest_appr_index].appr_count:
    #         highest_appr_index = i
    # print("Reference: img" + str(highest_appr_index))

convert_sdr2hdr(sdr_series)
cv2.waitKey(0)