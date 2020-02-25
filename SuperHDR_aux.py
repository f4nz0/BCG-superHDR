import numpy as np
import cv2

sat_threshold = 180
blk_threshold = 100


class sdrImage:
    def __init__(self, image):
        self.image = image
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        self.luminance = image[:,:,1].copy()
        self.trinarized = np.zeros(image[:,:,1].shape, dtype=np.uint8)
        self.pixels_appr = 0
        self.pixels_sat = 0
        self.pixels_blk = 0
        self.displacement = (0, 0)
        self.darker = None
        self.brighter = None

    def __str__(self):
        out = "Appropriate: " + str(self.pixels_appr) + "\n"
        out += "Blacked out: " + str(self.pixels_blk) + "\n"
        out += "Saturated: " + str(self.pixels_sat) + "\n"
        return out

def adjust_exposure(sdr_image, reference):
    image_mean = np.mean(sdr_image.luminance)
    # print(image_mean)
    ref_mean = np.mean(reference[:,:,1])
    ratio = ref_mean / image_mean
    multiplication = np.multiply(sdr_image.luminance, ratio)
    multiplication[multiplication > 255] = 255
    sdr_image.luminance = np.ndarray.astype(multiplication, dtype=np.uint8)
    # print(np.mean(sdr_image.luminance))
    # print(ref_mean)


def trinarize(sdrimage):
    luminance = sdrimage.luminance
    trinarized = np.ones(luminance.shape, dtype=np.uint8)
    trinarized += (luminance > sat_threshold)
    trinarized -= (luminance < blk_threshold)
    sdrimage.trinarized = trinarized


def count_pixels(sdrimage):
    # Returns (saturated, appropriate, blacked-out)
    sdrimage.pixels_sat = (sdrimage.luminance > sat_threshold).sum()
    sdrimage.pixels_blk = (sdrimage.luminance < blk_threshold).sum()
    sdrimage.pixels_appr = sdrimage.luminance.size - sdrimage.pixels_sat - sdrimage.pixels_blk


def trinarize_vis(tri):
    grayscale = np.zeros(tri.shape, dtype=np.uint8)
    grayscale[tri == 2] = 180
    grayscale[tri == 1] = 120
    grayscale[tri == 0] = 60
    return grayscale


def difference_mask(tri01, tri02):
    error = (tri01 != tri02).sum()
    return error


def difference_mask_vis(im01, im02):
    diff = (im01.trinarized != im02.trinarized) * 180
    diff = np.ndarray.astype(diff, dtype=np.uint8)
    diff[diff == 0] = 90
    return diff


def difference_mask_vis_aligned(im01, ref):
    x = im01.displacement[0]
    y = im01.displacement[1]
    alignment = np.zeros(ref.trinarized.shape, dtype=np.uint8) - 1
    image_tri = im01.trinarized
    if x < 0:
        align_x = (0, alignment.shape[0] + x)
        image_x = (-x, image_tri.shape[0])
    else:
        align_x = (x, image_tri.shape[0])
        image_x = (0, alignment.shape[0] - x)
    if y < 0:
        align_y = (0, alignment.shape[1] + y)
        image_y = (-y, image_tri.shape[1])
    else:
        align_y = (y, image_tri.shape[1])
        image_y = (0, alignment.shape[1] - y)
    diff = (im01.trinarized[align_x[0]: align_x[1], align_y[0]: align_y[1]] != ref.trinarized[image_x[0]: image_x[1], image_y[0]: image_y[1]]) * 180
    diff = np.ndarray.astype(diff, dtype=np.uint8)
    diff[diff == 0] = 90
    return diff


def scale_down(channel):
    resized = cv2.resize(channel, (int(channel.shape[1] * 0.1), int(channel.shape[0] * 0.1)), interpolation=cv2.INTER_NEAREST)
    return resized


def align_image(image_tri, ref_tri):
    rough_displ = align(scale_down(image_tri), scale_down(ref_tri))
    # print(rough_displ)
    displacement = align(image_tri, ref_tri, displacement=(rough_displ[0][0] * 10, rough_displ[0][1] * 10), previous_displacement=True)
    return displacement[0]


def align(image_tri, ref_tri, displacement=(0, 0), previous_displacement=False):
    displacements = []

    if not previous_displacement:
        ranges = [(-int(ref_tri.shape[0] * 0.10), int(ref_tri.shape[0] * 0.10)), (-int(ref_tri.shape[1] * 0.10), int(ref_tri.shape[1] * 0.10))]
    else:
        ranges = [(displacement[0] - 10, displacement[0] + 10), (displacement[1] - 10, displacement[1] + 10)]
    for x in range(ranges[0][0],ranges[0][1]):
        for y in range(ranges[1][0],ranges[1][1]):
            # print(x, y)
            alignment = np.zeros(ref_tri.shape, dtype=np.uint8) -1
            if x < 0:
                align_x = (0, alignment.shape[0] + x)
                image_x = (-x, image_tri.shape[0])
            else:
                align_x = (x, image_tri.shape[0])
                image_x = (0, alignment.shape[0] - x)
            if y < 0:
                align_y = (0, alignment.shape[1] + y)
                image_y = (-y, image_tri.shape[1])
            else:
                align_y = (y, image_tri.shape[1])
                image_y = (0, alignment.shape[1] - y)
            alignment[align_x[0]: align_x[1], align_y[0]: align_y[1]] = image_tri[image_x[0]: image_x[1], image_y[0]: image_y[1]]
            error = difference_mask(alignment, ref_tri)
            displacements.append(((x, y), error))
    smallest_error = displacements[0]
    for displacement in displacements:
        if displacement[1] < smallest_error[1]:
            smallest_error = displacement
    # print(smallest_error)
    return smallest_error


def sort_into_chain(ref, sdr_image):
    if sdr_image.pixels_blk > ref.pixels_blk:
        if ref.darker is None:
            ref.darker = sdr_image
            sdr_image.brighter = ref
        else:
            darker = ref.darker
            if darker.pixels_blk >= sdr_image.pixels_blk:
                sdr_image.brighter = ref
                sdr_image.darker = darker
                darker.brighter = sdr_image
                ref.darker = sdr_image
            else:
                sort_into_chain(darker, sdr_image)
    elif sdr_image.pixels_sat > ref.pixels_sat:
        if ref.brighter is None:
            ref.brighter = sdr_image
            sdr_image.darker = ref
        else:
            brighter = ref.brighter
            if brighter.pixels_sat >= sdr_image.pixels_sat:
                sdr_image.darker = ref
                sdr_image.brighter = brighter
                brighter.darker = sdr_image
                ref.brighter = sdr_image
            else:
                sort_into_chain(brighter, sdr_image)


def merging(ref):
    hdrimage = np.ones(ref.image.shape, dtype=np.uint8)
    for x in range(0, ref.image.shape[0]):
        for y in range(0, ref.image.shape[1]):
            temp_image = ref
            found = False
            # print(x , y)
            depth_counter = 0
            if temp_image.trinarized[x][y] == 2:
                while not found:
                    if temp_image.darker is not None:
                        depth_counter += 1
                        if temp_image.darker.trinarized[x][y] == 1:
                            temp_x = x + temp_image.displacement[0]
                            temp_y = y + temp_image.displacement[1]
                            if temp_image.image.shape[0] > temp_x > 0 and temp_image.image.shape[1] > temp_y > 0:
                                hdrimage[x][y][:] = temp_image.darker.image[temp_x][temp_y][:]
                                # print("Darker found")
                                found = True
                            else:
                                temp_image = temp_image.darker
                        else:
                            temp_image = temp_image.darker
                    else:
                        temp_x = x + temp_image.displacement[0]
                        temp_y = y + temp_image.displacement[1]
                        if temp_image.image.shape[0] > temp_x > 0 and temp_image.image.shape[1] > temp_y > 0:
                            hdrimage[x][y][:] = temp_image.image[temp_x][temp_y][:]
                            # print("Darker found")
                            found = True
            elif temp_image.trinarized[x][y] == 0:
                while not found:
                    if temp_image.brighter is not None:
                        depth_counter += 1
                        if temp_image.brighter.trinarized[x][y] == 1:
                            temp_x = x + temp_image.displacement[0]
                            temp_y = y + temp_image.displacement[1]
                            if temp_image.image.shape[0] > temp_x > 0 and temp_image.image.shape[1] > temp_y > 0:
                                hdrimage[x][y][:] = temp_image.brighter.image[temp_x][temp_y][:]
                                # print("Brighter found")
                                found = True
                            else:
                                temp_image = temp_image.brighter
                        else:
                            temp_image = temp_image.brighter
                    else:
                        temp_x = x + temp_image.displacement[0]
                        temp_y = y + temp_image.displacement[1]
                        if temp_image.image.shape[0] > temp_x > 0 and temp_image.image.shape[1] > temp_y > 0:
                            hdrimage[x][y][:] = temp_image.image[temp_x][temp_y][:]
                            # print("Brighter found")
                            found = True
            else:
                hdrimage[x][y][:] = temp_image.image[x][y][:]
    print("DONE")
    return hdrimage


def import_series(name):
    sdr_series = []
    if name == "parliament":
        sdr_series.append(sdrImage(cv2.imread("Parliament/The Parliament - ppw - 01.png", cv2.IMREAD_COLOR)))
        sdr_series.append(sdrImage(cv2.imread("Parliament/The Parliament - ppw - 02.png", cv2.IMREAD_COLOR)))
        sdr_series.append(sdrImage(cv2.imread("Parliament/The Parliament - ppw - 03.png", cv2.IMREAD_COLOR)))
        sdr_series.append(sdrImage(cv2.imread("Parliament/The Parliament - ppw - 04.png", cv2.IMREAD_COLOR)))
        sdr_series.append(sdrImage(cv2.imread("Parliament/The Parliament - ppw - 05.png", cv2.IMREAD_COLOR)))
        sdr_series.append(sdrImage(cv2.imread("Parliament/The Parliament - ppw - 06.png", cv2.IMREAD_COLOR)))
    elif name == "parliament2":
        # sdr_series.append(sdrImage(cv2.imread("Parliament_moved/01.png", cv2.IMREAD_COLOR)))
        sdr_series.append(sdrImage(cv2.imread("Parliament_moved/02.png", cv2.IMREAD_COLOR)))
        sdr_series.append(sdrImage(cv2.imread("Parliament_moved/03.png", cv2.IMREAD_COLOR)))
        sdr_series.append(sdrImage(cv2.imread("Parliament_moved/04.png", cv2.IMREAD_COLOR)))
        sdr_series.append(sdrImage(cv2.imread("Parliament_moved/05.png", cv2.IMREAD_COLOR)))
        # sdr_series.append(sdrImage(cv2.imread("Parliament_moved/06.png", cv2.IMREAD_COLOR)))
    elif name == "test":
        sdr_series.append(sdrImage(cv2.imread("Parliament/test1c.png", cv2.IMREAD_COLOR)))
        sdr_series.append(sdrImage(cv2.imread("Parliament/test2.png", cv2.IMREAD_COLOR)))
    return sdr_series