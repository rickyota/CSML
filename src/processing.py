import numpy as np
from cv2 import connectedComponents, connectedComponentsWithStats, watershed, dilate, erode, distanceTransform, DIST_L2, CC_STAT_AREA


# make sure im is only 0 and 255
def clean_im(im):
    im = im * 255
    im[im != 255] = 0
    return im


def postprocessing(im, thre_discard, wid_dilate):
    # discard <= thre_discard
    im = discard(im, thre_discard)
    # dilate wid_dilate
    im = dilate_im(im, wid_dilate)
    # erode wid_dilate
    im = erode_im(im, wid_dilate)
    # watershed
    im = watershed_im(im)
    return im


# discard isolated area under threshold
def discard(im, threshold):
    im = np.uint8(im)
    _, im_label, stats, _ = connectedComponentsWithStats(im, connectivity=4)

    mask = np.isin(im_label, np.where(stats[:, CC_STAT_AREA] <= threshold)[0])
    im[mask] = 0

    return im


# dilate width px
def dilate_im(im, width):
    neibor8 = np.ones((2 * width + 1, 2 * width + 1), np.uint8)
    im = dilate(im, neibor8, iterations=1)
    return im


# erode width px
def erode_im(im, width):
    neibor8 = np.ones((2 * width + 1, 2 * width + 1), np.uint8)
    im = erode(im, neibor8, iterations=1)
    return im


# extract lines
def watershed_im(im):
    im = np.asarray(im, np.uint8)
    im_dist = distanceTransform(im, DIST_L2, 5)
    im_dist = im_dist.reshape((im_dist.shape[0], im_dist.shape[1], -1))
    im_wsd = np.tile(im_dist, (1, 1, 3))
    im_wsd = np.asarray(im_wsd, np.uint8)
    im = 1 - im
    _, markers = connectedComponents(im)
    im_wsd = watershed(im_wsd, markers)
    im_wsd[0, :], im_wsd[-1, :], im_wsd[:, 0], im_wsd[:, -1] = 1, 1, 1, 1
    im_wsd = (im_wsd == -1)
    im_wsd = np.asarray(im_wsd, np.uint8)

    # fill 1px
    im_wsd = 1 - im_wsd
    _, im_label, stats, _ = connectedComponentsWithStats(
        im_wsd, connectivity=4)
    mask = np.isin(im_label, np.where(stats[:, CC_STAT_AREA] <= 1)[0])
    im_wsd = 1 - im_wsd
    im_wsd[mask] = 1

    return im_wsd
