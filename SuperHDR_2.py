import cv2
import numpy as np
from SuperHDR_aux import *


class sdrImage:
    def __init__(self, image):
        self.image = image
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        self.luminance = image[:,:,1]
        self.trinarized = np.zeros(image[:,:,1].shape, dtype=np.uint8)


im01 = sdrImage(cv2.imread("Parliament/test1.png", cv2.IMREAD_COLOR))
im02 = sdrImage(cv2.imread("Parliament/test2.png", cv2.IMREAD_COLOR))
# im01 = sdrImage(cv2.imread("Parliament/The Parliament - ppw - 04.png", cv2.IMREAD_COLOR))
# im02 = sdrImage(cv2.imread("Parliament/The Parliament - ppw - 05.png", cv2.IMREAD_COLOR))

cv2.imshow("01 before", im01.luminance)
adjust_exposure(im01, im02.image)
trinarize(im01)
trinarize(im02)

cv2.imshow("01", trinarize_vis(im01.trinarized))
cv2.imshow("02", trinarize_vis(im02.trinarized))
cv2.imshow("diff", difference_mask_vis(im01, im02))

align_image(im01.trinarized, im02.trinarized)


cv2.waitKey(0)