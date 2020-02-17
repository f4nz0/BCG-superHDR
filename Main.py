from SuperHDR import *

# image sources http://hdr-photographer.com/hdr-photos-to-play-with/


sdr_series = []
sdr_series.append(sdrImage(cv2.imread("Royal_Palace_in_Madrid/img_1335.jpg", cv2.IMREAD_COLOR)))
sdr_series.append(sdrImage(cv2.imread("Royal_Palace_in_Madrid/img_1336.jpg", cv2.IMREAD_COLOR)))
sdr_series.append(sdrImage(cv2.imread("Royal_Palace_in_Madrid/img_1337.jpg", cv2.IMREAD_COLOR)))
# sdr_series.append(sdrImage(cv2.imread("Royal_Palace_in_Madrid/img_1338.jpg", cv2.IMREAD_COLOR)))
# sdr_series.append(sdrImage(cv2.imread("Royal_Palace_in_Madrid/img_1339.jpg", cv2.IMREAD_COLOR)))
# sdr_series.append(sdrImage(cv2.imread("Royal_Palace_in_Madrid/img_1340.jpg", cv2.IMREAD_COLOR)))
# sdr_series.append(sdrImage(cv2.imread("Royal_Palace_in_Madrid/img_1341.jpg", cv2.IMREAD_COLOR)))






convert_sdr2hdr(sdr_series)
cv2.waitKey(0)