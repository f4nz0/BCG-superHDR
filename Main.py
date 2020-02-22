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

# sdr_series = []
# sdr_series.append(sdrImage(cv2.imread("shdr_1/shdr_1a.png", cv2.IMREAD_COLOR)))
# sdr_series.append(sdrImage(cv2.imread("shdr_1/shdr_1b.png", cv2.IMREAD_COLOR)))
# sdr_series.append(sdrImage(cv2.imread("shdr_1/shdr_1c.png", cv2.IMREAD_COLOR)))
# sdr_series.append(sdrImage(cv2.imread("shdr_1/shdr_1d.png", cv2.IMREAD_COLOR)))
# sdr_series.append(sdrImage(cv2.imread("shdr_1/shdr_1e.png", cv2.IMREAD_COLOR)))
# sdr_series.append(sdrImage(cv2.imread("shdr_1/shdr_1f.png", cv2.IMREAD_COLOR)))

# sdr_series = []
# sdr_series.append(sdrImage(cv2.imread("Parliament/The Parliament - ppw - 01.png", cv2.IMREAD_COLOR)))
# sdr_series.append(sdrImage(cv2.imread("Parliament/The Parliament - ppw - 02.png", cv2.IMREAD_COLOR)))
# sdr_series.append(sdrImage(cv2.imread("Parliament/The Parliament - ppw - 03.png", cv2.IMREAD_COLOR)))
# sdr_series.append(sdrImage(cv2.imread("Parliament/The Parliament - ppw - 04.png", cv2.IMREAD_COLOR)))
# sdr_series.append(sdrImage(cv2.imread("Parliament/The Parliament - ppw - 05.png", cv2.IMREAD_COLOR)))
# sdr_series.append(sdrImage(cv2.imread("Parliament/The Parliament - ppw - 06.png", cv2.IMREAD_COLOR)))
#
# maxval = 0
# for i in range(0, len(sdr_series)):
#     for x in range(0, sdr_series[i].image.shape[0]):
#         for y in range(0, sdr_series[i].image.shape[1]):
#             if sdr_series[i].image[x][y][1] > maxval:
#                 maxval = sdr_series[i].image[x][y][1]
#     print(maxval)

convert_sdr2hdr(sdr_series)
cv2.waitKey(0)