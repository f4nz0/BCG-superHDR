import cv2
import numpy as np

# image sources http://hdr-photographer.com/hdr-photos-to-play-with/

sat_threshold = 200
blk_threshold = 60




def merging(ref):
    hdrimage = ref.image.copy()
    # hdrimage = np.zeros(ref.image.shape, dtype=np.uint8)
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
                            if temp.darker.trinarized[new_x][new_y] == 1:
                                hdrimage[x][y][:] = merge_pixels(hdrimage[x][y][:], temp.darker.image[new_x][new_y][:], depth_counter)
                                found = True
                                # print("replaced")
                            else:
                                hdrimage[x][y][:] = merge_pixels(hdrimage[x][y][:], temp.darker.image[new_x][new_y][:], depth_counter)

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
                            if temp.brighter.trinarized[new_x][new_y] == 1:
                                hdrimage[x][y][:] = merge_pixels(hdrimage[x][y][:],
                                                                 temp.brighter.image[new_x][new_y][:],
                                                                 depth_counter)
                                found = True
                                # print("replaced")
                            else:
                                hdrimage[x][y][:] = merge_pixels(hdrimage[x][y][:],temp.brighter.image[new_x][new_y][:],depth_counter)
                                temp = temp.brighter
                        else:
                            temp = temp.brighter
                    else:
                        found = True
    # hdrimage = cv2.blur(hdrimage, (2, 2))
    # hdrimage = replace(hdrimage, ref)
    # hdrimage = np.ndarray.astype(hdrimage, dtype=np.uint8)
    return hdrimage


def merge_pixels(pixel, new_pixel, weight):
    return (pixel * (1 - (1 / weight))) + (new_pixel * (1/weight))

def merging(ref):
    hdrimage = ref.image.copy()
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
                            if temp.darker.trinarized[new_x][new_y] == 1:
                                hdrimage[x][y][:] = get_pixel_value(temp.darker.image[x][y][:], 2, temp.darker.relative_exp, ref.relative_exp)
                                found = True
                                # print("replaced")
                            else:
                                hdrimage[x][y][:] = get_pixel_value(temp.darker.image[x][y][:], 2, temp.darker.relative_exp, ref.relative_exp)

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
                            if temp.brighter.trinarized[new_x][new_y] == 1:
                                hdrimage[x][y][:] = get_pixel_value(temp.brighter.image[x][y][:], 0, temp.brighter.relative_exp, ref.relative_exp)
                                found = True
                                # print("replaced")
                            else:
                                hdrimage[x][y][:] = get_pixel_value(temp.brighter.image[x][y][:], 0, temp.brighter.relative_exp, ref.relative_exp)
                                temp = temp.brighter
                        else:
                            temp = temp.brighter
                    else:
                        found = True
    return hdrimage


def get_pixel_value(pixel, tri, exposure_image, exposure_ref):
    val_at_exp = (1/(1+np.e ** (-exposure_image * 0.65))) * 255
    val_at_ref = (1/(1+np.e ** (-exposure_ref * 0.65))) * 255

    if tri == 2:
        diff = np.absolute(val_at_ref - val_at_exp)
        # print(diff)
        pixel_lum = pixel[1] + diff
    if tri == 0:
        diff = np.absolute(val_at_exp - val_at_ref)
        # print(diff)
        pixel_lum = pixel[1] - diff

    # pixel_lum = pixel[1] + diff
    if pixel_lum > 255:
        pixel_lum = 255
    elif pixel_lum < 0:
        pixel_lum = 0
    new_pixel = [pixel[0], pixel_lum, pixel[2]]
    return new_pixel




























def merging(ref):
    hdrimage = ref.image.copy() / 255
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
                            if temp.darker.trinarized[new_x][new_y] == 1:
                                hdrimage[x][y][:] = get_pixel_value(temp.darker.image[x][y][:], 2, temp.darker.relative_exp, ref.relative_exp)
                                found = True
                                # print("replaced")
                            else:
                                hdrimage[x][y][:] = get_pixel_value(temp.darker.image[x][y][:], 2, temp.darker.relative_exp, ref.relative_exp)

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
                            if temp.brighter.trinarized[new_x][new_y] == 1:
                                hdrimage[x][y][:] = get_pixel_value(temp.brighter.image[x][y][:], 0, temp.brighter.relative_exp, ref.relative_exp)
                                found = True
                                # print("replaced")
                            else:
                                hdrimage[x][y][:] = get_pixel_value(temp.brighter.image[x][y][:], 0, temp.brighter.relative_exp, ref.relative_exp)
                                temp = temp.brighter
                        else:
                            temp = temp.brighter
                    else:
                        found = True
    hdrimage = np.ndarray.astype(hdrimage, np.float32)
    # hdrimage = cv2.cvtColor(hdrimage, cv2.COLOR_HLS2BGR)
    return hdrimage


def get_pixel_value(pixel, tri, exposure_image, exposure_ref):
    pixel = pixel / 255
    val_at_exp = (1/(1+np.e ** (-exposure_image)))
    val_at_ref = (1/(1+np.e ** (-exposure_ref)))
    if tri == 2:
        diff = np.absolute(val_at_ref - val_at_exp)
        pixel_lum = pixel[1] + diff
    if tri == 0:
        diff = np.absolute(val_at_exp - val_at_ref)
        pixel_lum = pixel[1] - diff
    new_pixel = [pixel[0], pixel_lum, pixel[2]]