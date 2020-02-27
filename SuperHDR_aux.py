import numpy as np
import cv2

sat_threshold = 190
blk_threshold = 90


class sdrImage:
    def __init__(self, image, exp):
        # The image in BGR color space
        self.image = image
        # Temporary conversion to HSV for getting the brightness channel
        imagetemp = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        self.luminance = imagetemp[:,:,2]
        self.trinarized = np.zeros(self.luminance.shape, dtype=np.uint8)
        # Amount of appropriate pixels
        self.pixels_appr = 0
        self.pixels_sat = 0
        self.pixels_blk = 0
        self.displacement = (0, 0)
        self.darker = None
        self.brighter = None
        self.relative_exp = exp

    def __str__(self):
        out = "Appropriate: " + str(self.pixels_appr) + "\n"
        out += "Blacked out: " + str(self.pixels_blk) + "\n"
        out += "Saturated: " + str(self.pixels_sat) + "\n"
        return out

# Calculates the exposure ratio between the image and the reference image,
# then changes the image's luminance according to the ratio to roughly achieve similar brightness
def adjust_exposure(sdr_image, reference):
    image_mean = np.mean(sdr_image.luminance)
    # print(image_mean)
    ref_mean = np.mean(reference.luminance)
    ratio = ref_mean / image_mean
    multiplication = np.multiply(sdr_image.luminance, ratio)
    multiplication[multiplication > 255] = 255
    sdr_image.luminance = np.ndarray.astype(multiplication, dtype=np.uint8)
    # print(np.mean(sdr_image.luminance))
    # print(ref_mean)


# Takes the luminance channel and outputs an array with one value per pixel
# The value is 2 if the pixel is brighter than the threshold
# The value is 0 if the pixel is darker than the threshold
# The value is 1 if the pixel is neither of those, which is to say appropriate
def trinarize(sdrimage):
    luminance = sdrimage.luminance
    trinarized = np.ones(luminance.shape, dtype=np.uint8)
    trinarized += (luminance > sat_threshold)
    trinarized -= (luminance < blk_threshold)
    sdrimage.trinarized = trinarized


# Calculates the amount of saturated, blacked-out and appropriate pixels in the image acc. to thresholds
def count_pixels(sdrimage):
    sdrimage.pixels_sat = (sdrimage.luminance > sat_threshold).sum()
    sdrimage.pixels_blk = (sdrimage.luminance < blk_threshold).sum()
    sdrimage.pixels_appr = sdrimage.luminance.size - sdrimage.pixels_sat - sdrimage.pixels_blk


# Outputs a grayscale visual interpretation of the trinarization
def trinarize_vis(tri):
    grayscale = np.zeros(tri.shape, dtype=np.uint8)
    grayscale[tri == 2] = 230
    grayscale[tri == 1] = 120
    grayscale[tri == 0] = 40
    return grayscale

# Calculates how many pixels align between the two images
def difference_mask(tri01, tri02):
    # When using displacements, some pixels on the overlay don't exist
    # These are noted as 3 in the trinarized image
    # ignore_pixels counts them out
    ignore_pixels = tri01[tri01 == 3].sum() / 3
    error = (tri01 != tri02).sum() - ignore_pixels
    # print(error)
    return error

# Outputs a visual representation of the difference mask
def difference_mask_vis(im01, im02):
    diff = (im01.trinarized != im02.trinarized) * 180
    diff = np.ndarray.astype(diff, dtype=np.uint8)
    diff[diff == 0] = 90
    return diff


# Scales down the image to a tenth of its size
def scale_down(channel):
    resized = cv2.resize(channel, (int(channel.shape[1] * 0.1), int(channel.shape[0] * 0.1)), interpolation=cv2.INTER_NEAREST)
    return resized

# Aligns the images. If the image is quite big, an initial alignment is done on a downscaled version.
# After that, the alignment of the original image can be down with the downscaled alignment as a starting point
def align_image(image_tri, ref_tri):
    if image_tri.shape[0] > 800:
        rough_displ = align(scale_down(image_tri), scale_down(ref_tri))
        # print(rough_displ)
        displacement = align(image_tri, ref_tri, displacement=(rough_displ[0][0] * 10, rough_displ[0][1] * 10), previous_displacement=True)
        return displacement[0]
    else:
        displacement = align(image_tri, ref_tri)
        return displacement[0]


def align(image_tri, ref_tri, displacement=(0, 0), previous_displacement=False):
    displacements = []
    # If the image has not been displaced before, the displacement is done with a maximum of 10% to either side
    # If the image was displaced before (at a tenth of its size), the algorithm only needs to check +/- 10px to either side
    # of the downscaled displacement
    if not previous_displacement:
        ranges = [(-int(ref_tri.shape[0] * 0.10), int(ref_tri.shape[0] * 0.10)), (-int(ref_tri.shape[1] * 0.10), int(ref_tri.shape[1] * 0.10))]
    else:
        ranges = [(displacement[0] - 10, displacement[0] + 10), (displacement[1] - 10, displacement[1] + 10)]
    for x in range(ranges[0][0],ranges[0][1]):
        for y in range(ranges[1][0],ranges[1][1]):
            # print(x, y)
            alignment = np.zeros(ref_tri.shape, dtype=np.uint8) + 3
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


# Sorts the images into a chain from darkest to brightest with the reference image in between
# This allows for the merging algorithm to find the next-darkest image of e.g. the reference image at reference.darker
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


# Starts with an empty HDR image and checks every pixel
# If the pixel in the reference image is appropriate, its radiance is noted in the HDR image
# If the pixel is saturated, starting from the reference, the algorithm looks at the next-darkest image
# to find an appropriate pixel. If found, it notes its radiance in the HDR image. If not, it goes to the next-darkest image.
# The same is done for blacked-out pixels (i.e. analyzing all the brighter images)
def merging(ref):
    hdrimage = np.zeros(ref.image.shape, dtype=np.float32)
    for x in range(0, hdrimage.shape[0]):
        for y in range(0, hdrimage.shape[1]):
            temp = ref
            found = False
            depth_counter = 0
            if ref.trinarized[x][y] == 2:
                # print("replacing", x, y)
                while not found:
                    if temp.darker is not None:
                        depth_counter += 1
                        displ = temp.darker.displacement
                        new_x = x - displ[0]
                        new_y = y - displ[1]
                        if temp.darker.image.shape[0] > new_x > 0 and temp.darker.image.shape[1] > new_y > 0:
                            hdrimage[x][y][:] = inv_crf(temp.darker.image[new_x][new_y][:], temp.darker.relative_exp, ref.relative_exp)
                            if temp.darker.trinarized[new_x][new_y] == 1:
                                found = True
                            else:
                                temp = temp.darker
                        else:
                            temp = temp.darker
                    else:
                        found = True
            elif ref.trinarized[x][y] == 0:
                while not found:
                    if temp.brighter is not None:
                        depth_counter += 1
                        displ = temp.brighter.displacement
                        new_x = x - displ[0]
                        new_y = y - displ[1]
                        if temp.brighter.image.shape[0] > new_x > 0 and temp.brighter.image.shape[1] > new_y > 0:
                            hdrimage[x][y][:] = inv_crf(temp.brighter.image[new_x][new_y][:], temp.brighter.relative_exp, ref.relative_exp)
                            if temp.brighter.trinarized[new_x][new_y] == 1:
                                found = True
                            else:
                                temp = temp.brighter
                        else:
                            temp = temp.brighter
                    else:
                        found = True
            else:
                hdrimage[x][y][:] = inv_crf(ref.image[x][y][:], 0, 0)
    return hdrimage


# Calculates the inverse Camera Response Function for an irradiance value (pixel value).
# Takes the relation between the exposure of the reference pixel and the new pixel and shifts the radiance
# of the new pixel, so that it aligns with the reference
def inv_crf(pixel, exposure_image, exposure_ref):
    shift = exposure_image - exposure_ref
    exposure_values = pixel.copy()
    exposure_values = np.ndarray.astype(exposure_values, dtype=np.float32)
    pixel[pixel == 0] = 1
    pixel[pixel == 255] = 254
    exposure_values[0] = 2**(-1.5*np.log((255/pixel[0]) - 1) - shift)
    exposure_values[1] = 2**(-1.5*np.log((255/pixel[1]) - 1) - shift)
    exposure_values[2] = 2**(-1.5*np.log((255/pixel[2]) - 1) - shift)
    exposure_values[exposure_values > 5000.0] = 5000
    exposure_values[exposure_values < -0] = 0
    return exposure_values


# Easy importing of all the images that were used in this project
def import_series(name):
    sdr_series = []
    if name == "parliament": # resource: https://farbspiel-photo.com/learn/hdr-pics-to-play-with/hdr-pics-to-play-with-the-parliament
        sdr_series.append(sdrImage(cv2.imread("Parliament/The Parliament - ppw - 01.png", cv2.IMREAD_COLOR), -3))
        sdr_series.append(sdrImage(cv2.imread("Parliament/The Parliament - ppw - 02.png", cv2.IMREAD_COLOR), -2))
        sdr_series.append(sdrImage(cv2.imread("Parliament/The Parliament - ppw - 03.png", cv2.IMREAD_COLOR), -1))
        sdr_series.append(sdrImage(cv2.imread("Parliament/The Parliament - ppw - 04.png", cv2.IMREAD_COLOR), 0))
        sdr_series.append(sdrImage(cv2.imread("Parliament/The Parliament - ppw - 05.png", cv2.IMREAD_COLOR), 1))
        sdr_series.append(sdrImage(cv2.imread("Parliament/The Parliament - ppw - 06.png", cv2.IMREAD_COLOR), 2))
    elif name == "own_street": # Photos were captured by us!
        # sdr_series.append(sdrImage(cv2.imread("own/street_0.JPG", cv2.IMREAD_COLOR), -6))
        # sdr_series.append(sdrImage(cv2.imread("own/street_1.JPG", cv2.IMREAD_COLOR), -5))
        # sdr_series.append(sdrImage(cv2.imread("own/street_2.JPG", cv2.IMREAD_COLOR), -4))
        # sdr_series.append(sdrImage(cv2.imread("own/street_3.JPG", cv2.IMREAD_COLOR), -3))
        sdr_series.append(sdrImage(cv2.imread("own/street_4.JPG", cv2.IMREAD_COLOR), -2))
        sdr_series.append(sdrImage(cv2.imread("own/street_5.JPG", cv2.IMREAD_COLOR), -1))
        sdr_series.append(sdrImage(cv2.imread("own/street_6.JPG", cv2.IMREAD_COLOR), 0))
        sdr_series.append(sdrImage(cv2.imread("own/street_7.JPG", cv2.IMREAD_COLOR), 1))
        sdr_series.append(sdrImage(cv2.imread("own/street_8.JPG", cv2.IMREAD_COLOR), 2))
        # sdr_series.append(sdrImage(cv2.imread("own/street_9.JPG", cv2.IMREAD_COLOR), 3))
    elif name == "own_statue": # Photos were captured by us!
        # sdr_series.append(sdrImage(cv2.imread("own/statue_0.JPG", cv2.IMREAD_COLOR), -6))
        # sdr_series.append(sdrImage(cv2.imread("own/statue_1.JPG", cv2.IMREAD_COLOR), -5))
        sdr_series.append(sdrImage(cv2.imread("own/statue_2.JPG", cv2.IMREAD_COLOR), -4))
        sdr_series.append(sdrImage(cv2.imread("own/statue_3.JPG", cv2.IMREAD_COLOR), -3))
        sdr_series.append(sdrImage(cv2.imread("own/statue_4.JPG", cv2.IMREAD_COLOR), -2))
        sdr_series.append(sdrImage(cv2.imread("own/statue_5.JPG", cv2.IMREAD_COLOR), -1))
        sdr_series.append(sdrImage(cv2.imread("own/statue_6.JPG", cv2.IMREAD_COLOR), 0))
        sdr_series.append(sdrImage(cv2.imread("own/statue_7.JPG", cv2.IMREAD_COLOR), 1))
        sdr_series.append(sdrImage(cv2.imread("own/statue_8.JPG", cv2.IMREAD_COLOR), 2))
        sdr_series.append(sdrImage(cv2.imread("own/statue_9.JPG", cv2.IMREAD_COLOR), 3))
    elif name == "own_street_moved": # Photos were captured by us!
        # sdr_series.append(sdrImage(cv2.imread("own/street2 (1).JPG", cv2.IMREAD_COLOR), -5))
        # sdr_series.append(sdrImage(cv2.imread("own/street2 (2).JPG", cv2.IMREAD_COLOR), -4))
        # sdr_series.append(sdrImage(cv2.imread("own/street2 (3).JPG", cv2.IMREAD_COLOR), -3))
        sdr_series.append(sdrImage(cv2.imread("own/street2 (4).JPG", cv2.IMREAD_COLOR), -2))
        sdr_series.append(sdrImage(cv2.imread("own/street2 (5).JPG", cv2.IMREAD_COLOR), -1))
        sdr_series.append(sdrImage(cv2.imread("own/street2 (6).JPG", cv2.IMREAD_COLOR), 0))
        sdr_series.append(sdrImage(cv2.imread("own/street2 (7).JPG", cv2.IMREAD_COLOR), 1))
        sdr_series.append(sdrImage(cv2.imread("own/street2 (8).JPG", cv2.IMREAD_COLOR), 2))
        # sdr_series.append(sdrImage(cv2.imread("own/street2 (9).JPG", cv2.IMREAD_COLOR), 3))
    elif name == "own_street_moved_enhanced": # Photos were captured by us!
        # sdr_series.append(sdrImage(cv2.imread("own/street2b (1).JPG", cv2.IMREAD_COLOR), -5))
        # sdr_series.append(sdrImage(cv2.imread("own/street2b (2).JPG", cv2.IMREAD_COLOR), -4))
        # sdr_series.append(sdrImage(cv2.imread("own/street2b (3).JPG", cv2.IMREAD_COLOR), -3))
        sdr_series.append(sdrImage(cv2.imread("own/street2b (4).JPG", cv2.IMREAD_COLOR), -2))
        sdr_series.append(sdrImage(cv2.imread("own/street2b (5).JPG", cv2.IMREAD_COLOR), -1))
        sdr_series.append(sdrImage(cv2.imread("own/street2b (6).JPG", cv2.IMREAD_COLOR), 0))
        # sdr_series.append(sdrImage(cv2.imread("own/street2b (7).JPG", cv2.IMREAD_COLOR), 1))
        sdr_series.append(sdrImage(cv2.imread("own/street2b (8).JPG", cv2.IMREAD_COLOR), 2))
        # sdr_series.append(sdrImage(cv2.imread("own/street2b (9).JPG", cv2.IMREAD_COLOR), 3))
    elif name == "own_street_moved_mini": # Photos were captured by us!
        # sdr_series.append(sdrImage(cv2.imread("own/mini/street2b (1).JPG", cv2.IMREAD_COLOR), -5))
        # sdr_series.append(sdrImage(cv2.imread("own/mini/street2b (2).JPG", cv2.IMREAD_COLOR), -4))
        # sdr_series.append(sdrImage(cv2.imread("own/mini/street2b (3).JPG", cv2.IMREAD_COLOR), -3))
        sdr_series.append(sdrImage(cv2.imread("own/mini/street2b (4).JPG", cv2.IMREAD_COLOR), -2))
        sdr_series.append(sdrImage(cv2.imread("own/mini/street2b (5).JPG", cv2.IMREAD_COLOR), -1))
        sdr_series.append(sdrImage(cv2.imread("own/mini/street2b (6).JPG", cv2.IMREAD_COLOR), 0))
        # sdr_series.append(sdrImage(cv2.imread("own/mini/street2b (7).JPG", cv2.IMREAD_COLOR), 1))
        sdr_series.append(sdrImage(cv2.imread("own/mini/street2b (8).JPG", cv2.IMREAD_COLOR), 2))
        sdr_series.append(sdrImage(cv2.imread("own/mini/street2b (9).JPG", cv2.IMREAD_COLOR), 3))
    elif name == "parliament2": # resource: https://farbspiel-photo.com/learn/hdr-pics-to-play-with/hdr-pics-to-play-with-the-parliament
        # sdr_series.append(sdrImage(cv2.imread("Parliament_moved/01.png", cv2.IMREAD_COLOR), -3))
        sdr_series.append(sdrImage(cv2.imread("Parliament_moved/02.png", cv2.IMREAD_COLOR),-2))
        sdr_series.append(sdrImage(cv2.imread("Parliament_moved/03.png", cv2.IMREAD_COLOR),-1))
        sdr_series.append(sdrImage(cv2.imread("Parliament_moved/04.png", cv2.IMREAD_COLOR),0))
        sdr_series.append(sdrImage(cv2.imread("Parliament_moved/05.png", cv2.IMREAD_COLOR),1))
        # sdr_series.append(sdrImage(cv2.imread("Parliament_moved/06.png", cv2.IMREAD_COLOR),2))
    elif name == "parliament2mini": # resource: https://farbspiel-photo.com/learn/hdr-pics-to-play-with/hdr-pics-to-play-with-the-parliament
        sdr_series.append(sdrImage(cv2.imread("Parliament_moved/mini/01.png", cv2.IMREAD_COLOR),-3))
        sdr_series.append(sdrImage(cv2.imread("Parliament_moved/mini/02.png", cv2.IMREAD_COLOR),-2))
        sdr_series.append(sdrImage(cv2.imread("Parliament_moved/mini/03.png", cv2.IMREAD_COLOR),-1))
        sdr_series.append(sdrImage(cv2.imread("Parliament_moved/mini/04.png", cv2.IMREAD_COLOR),0))
        sdr_series.append(sdrImage(cv2.imread("Parliament_moved/mini/05.png", cv2.IMREAD_COLOR),1))
        # sdr_series.append(sdrImage(cv2.imread("Parliament_moved/mini/06.png", cv2.IMREAD_COLOR),2))
    elif name == "opencv_test_set": # resource: https://github.com/opencv/opencv_extra/tree/3.4/testdata/cv/hdr/exposures
        sdr_series.append(sdrImage(cv2.imread("opencvhdr/memorial00.png", cv2.IMREAD_COLOR), 2))
        sdr_series.append(sdrImage(cv2.imread("opencvhdr/memorial01.png", cv2.IMREAD_COLOR), 1))
        sdr_series.append(sdrImage(cv2.imread("opencvhdr/memorial02.png", cv2.IMREAD_COLOR), 0))
        sdr_series.append(sdrImage(cv2.imread("opencvhdr/memorial03.png", cv2.IMREAD_COLOR), -1))
        sdr_series.append(sdrImage(cv2.imread("opencvhdr/memorial04.png", cv2.IMREAD_COLOR), -2))
        sdr_series.append(sdrImage(cv2.imread("opencvhdr/memorial05.png", cv2.IMREAD_COLOR), -3))
        sdr_series.append(sdrImage(cv2.imread("opencvhdr/memorial06.png", cv2.IMREAD_COLOR), -4))
        sdr_series.append(sdrImage(cv2.imread("opencvhdr/memorial07.png", cv2.IMREAD_COLOR), -5))
        sdr_series.append(sdrImage(cv2.imread("opencvhdr/memorial08.png", cv2.IMREAD_COLOR), -6))
        sdr_series.append(sdrImage(cv2.imread("opencvhdr/memorial09.png", cv2.IMREAD_COLOR), -7))
        # sdr_series.append(sdrImage(cv2.imread("opencvhdr/memorial10.png", cv2.IMREAD_COLOR), -8))
        # sdr_series.append(sdrImage(cv2.imread("opencvhdr/memorial11.png", cv2.IMREAD_COLOR), -9))
        # sdr_series.append(sdrImage(cv2.imread("opencvhdr/memorial12.png", cv2.IMREAD_COLOR), -10))
        # sdr_series.append(sdrImage(cv2.imread("opencvhdr/memorial13.png", cv2.IMREAD_COLOR), -11))
        # sdr_series.append(sdrImage(cv2.imread("opencvhdr/memorial14.png", cv2.IMREAD_COLOR)))
        # sdr_series.append(sdrImage(cv2.imread("opencvhdr/memorial15.png", cv2.IMREAD_COLOR)))
    elif name == "own_monster": # Photos were captured by us!
        sdr_series.append(sdrImage(cv2.imread("own/monster (1).JPG", cv2.IMREAD_COLOR), -6))
        sdr_series.append(sdrImage(cv2.imread("own/monster (2).JPG", cv2.IMREAD_COLOR), -5))
        sdr_series.append(sdrImage(cv2.imread("own/monster (3).JPG", cv2.IMREAD_COLOR), -4))
        sdr_series.append(sdrImage(cv2.imread("own/monster (4).JPG", cv2.IMREAD_COLOR), -3))
        sdr_series.append(sdrImage(cv2.imread("own/monster (5).JPG", cv2.IMREAD_COLOR), -2))
        sdr_series.append(sdrImage(cv2.imread("own/monster (6).JPG", cv2.IMREAD_COLOR), -1))
        sdr_series.append(sdrImage(cv2.imread("own/monster (7).JPG", cv2.IMREAD_COLOR), 0))
        sdr_series.append(sdrImage(cv2.imread("own/monster (8).JPG", cv2.IMREAD_COLOR), 1))
        sdr_series.append(sdrImage(cv2.imread("own/monster (9).JPG", cv2.IMREAD_COLOR), 2))
    elif name == "own_monster_mini": # Photos were captured by us!
        # sdr_series.append(sdrImage(cv2.imread("own/mini/monster (1).JPG", cv2.IMREAD_COLOR), -6))
        # sdr_series.append(sdrImage(cv2.imread("own/mini/monster (2).JPG", cv2.IMREAD_COLOR), -5))
        # sdr_series.append(sdrImage(cv2.imread("own/mini/monster (3).JPG", cv2.IMREAD_COLOR), -4))
        sdr_series.append(sdrImage(cv2.imread("own/mini/monster (4).JPG", cv2.IMREAD_COLOR), -3))
        sdr_series.append(sdrImage(cv2.imread("own/mini/monster (5).JPG", cv2.IMREAD_COLOR), -2))
        sdr_series.append(sdrImage(cv2.imread("own/mini/monster (6).JPG", cv2.IMREAD_COLOR), -1))
        sdr_series.append(sdrImage(cv2.imread("own/mini/monster (7).JPG", cv2.IMREAD_COLOR), 0))
        sdr_series.append(sdrImage(cv2.imread("own/mini/monster (8).JPG", cv2.IMREAD_COLOR), 1))
        sdr_series.append(sdrImage(cv2.imread("own/mini/monster (9).JPG", cv2.IMREAD_COLOR), 2))
    return sdr_series
