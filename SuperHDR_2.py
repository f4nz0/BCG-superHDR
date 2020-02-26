import cv2
import numpy as np
from SuperHDR_aux import *


def preprocess(sdr_series):
    ref = sdr_series[0]
    for sdr_image in sdr_series:
        count_pixels(sdr_image)
        if ref.pixels_appr < sdr_image.pixels_appr:
            ref = sdr_image
    for sdr_image in sdr_series:
        sort_into_chain(ref, sdr_image)
    # temp = ref
    # output = str(ref)
    # while temp.darker is not None:
    #     output = str(temp.darker) + output
    #     temp = temp.darker
    # temp = ref
    # while temp.brighter is not None:
    #     output += str(temp.brighter)
    #     temp = temp.brighter
    # print(output)
    return ref


def normalize_luminance(sdr_series, ref):
    for sdr_image in sdr_series:
        if sdr_image != ref:
            adjust_exposure(sdr_image, ref.image)
            # sdr_image.image[:,:,1] = sdr_image.luminance


def align_images_to_ref(sdr_series, ref):
    trinarize(ref)
    cv2.imshow("ref tri", trinarize_vis(ref.trinarized))
    for sdr_image in sdr_series:
        if sdr_image != ref:
            trinarize(sdr_image)
            sdr_image.displacement = align_image(sdr_image.trinarized, ref.trinarized)
            print(sdr_image.displacement)


def sdr_series_to_hdr(sdr_series):
    ref = preprocess(sdr_series)
    # cv2.imshow("ref", cv2.cvtColor(ref.image, cv2.COLOR_HLS2BGR))
    normalize_luminance(sdr_series, ref)
    align_images_to_ref(sdr_series, ref)
    hdr = merging(ref)
    cv2.imshow("hdr", cv2.cvtColor(hdr, cv2.COLOR_HLS2BGR))
    cv2.imshow("ref", cv2.cvtColor(ref.image, cv2.COLOR_HLS2BGR))
    # diff = (hdr != ref.image) * 120
    # cv2.imshow("diff", diff)


sdr_series = import_series("opencv_test_set")
sdr_series_to_hdr(sdr_series)


cv2.waitKey(0)
