import numpy as np
import cv2

sat_threshold = 220
blk_threshold = 140


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


def scale_down(channel):
    resized = cv2.resize(channel, (int(channel.shape[1] * 0.1), int(channel.shape[0] * 0.1)), interpolation=cv2.INTER_CUBIC)
    return resized


def align_image(image_tri, ref_tri):
    rough_displ = align(scale_down(image_tri), scale_down(ref_tri))
    print(rough_displ)
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
            print(x, y)
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
    print(smallest_error)
    return smallest_error
