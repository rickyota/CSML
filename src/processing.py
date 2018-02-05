import numpy as np
from cv2 import connectedComponents, connectedComponentsWithStats, watershed, dilate, distanceTransform, DIST_L2, CC_STAT_AREA
from epoch import infer_epoch


# infer a whole image
def infer_im(model, cim, im, thre_discard, wid_dilate, thre_fill):
    im = combine_im(model, cim, im)

    im = postprocessing(im, thre_discard, wid_dilate, thre_fill)
    im = clean_im(im)
    im = np.asarray(im, np.uint8)

    return im


# make sure im is only 0 and 255
def clean_im(im):
    im = im * 255
    im[im != 255] = 0
    return im


# get image after adapting max of two
def combine_im(model, cim, im):
    shape_ori = im.shape
    if im.shape[0] % cim.hgh != 0:
        im = np.lib.pad(
            im, ((0, cim.hgh - im.shape[0] % cim.hgh), (0, 0)), 'edge')
    if im.shape[1] % cim.wid != 0:
        im = np.lib.pad(
            im, ((0, 0), (0, cim.wid - im.shape[1] % cim.wid)), 'edge')

    iml = each_im(model, cim, im, 'l')
    ims = each_im(model, cim, im, 's')

    im = np.maximum(iml, ims)
    im = im[0:shape_ori[0], 0:shape_ori[1]]

    return im


# get image directly output from classifier
def each_im(model, cim, im, imtype):
    if imtype == 'l':
        d = 0
    elif imtype == 's':
        d = 1

    x_batch = []
    x_result = []
    k = 0
    for i in range(int((d / 2) * cim.hgh), im.shape[0] - cim.hgh + 1, cim.hgh):
        for j in range(int((d / 2) * cim.wid), im.shape[1] - cim.wid + 1, cim.wid):
            x_batch.append(im[i:i + cim.hgh, j:j + cim.wid])

            k += 1
            if k % 100 == 0:
                x_batch = np.asarray(x_batch, np.float32)
                x_batch = x_batch.reshape([-1] + cim.shapex)
                x_result_t = infer_epoch(model, x_batch)
                x_result.extend(x_result_t)
                x_batch = []

    if k % 100 != 0:
        x_batch = np.asarray(x_batch, np.float32)
        x_batch = x_batch.reshape([-1] + cim.shapex)
        x_result_t = infer_epoch(model, x_batch)
        x_result.extend(x_result_t)

    im_inferred = np.zeros_like(im)
    k = 0
    for i in range(int((d / 2) * cim.hgh), im.shape[0] - cim.hgh + 1, cim.hgh):
        for j in range(int((d / 2) * cim.wid), im.shape[1] - cim.wid + 1, cim.wid):
            im_inferred[i:i + cim.hgh, j:j + cim.wid] = x_result[k]
            k += 1

    return im_inferred


# get last image
def postprocessing(im, thre_discard, wid_dilate, thre_fill):
    # discard <= thre_discard
    im = discard(im, thre_discard)
    # dilate wid_dilate
    im = dilate_im(im, wid_dilate)
    # fill <= thre_fill
    im = fill(im, thre_fill)
    # watershed
    im = watershed_im(im)
    return im


# discard isolated area under threshold
def discard(im, threshold):
    im = np.uint8(im)
    n, im_label, stats, _ = connectedComponentsWithStats(im, connectivity=4)
    # OK? added np.int8
    im_discard = np.zeros_like(im, np.uint8)
    for i in range(1, n):
        if stats[i, CC_STAT_AREA] > threshold:
            im_discard[im_label == i] = 1
    return im_discard


# dilate width px
def dilate_im(im, width):
    neibor8 = np.ones((width + 2, width + 2), np.uint8)
    im = dilate(im, neibor8, iterations=1)
    return im


# fill area under threshld
def fill(im, threshold):
    im = 1 - im
    n, im_label, stats, _ = connectedComponentsWithStats(im, connectivity=4)
    for i in range(1, n):
        if stats[i, CC_STAT_AREA] <= threshold:
            im[im_label == i] = 0
    return im


# extract lines
def watershed_im(im):
    im = np.asarray(im, np.uint8)
    im = 1 - im
    im_dist = distanceTransform(im, DIST_L2, 5)
    im_dist = im_dist.reshape((im_dist.shape[0], im_dist.shape[1], -1))
    im_wsd = np.tile(im_dist, (1, 1, 3))
    im_wsd = np.asarray(im_wsd, np.uint8)
    im = 1 - im
    _, markers = connectedComponents(im)
    im_wsd = watershed(im_wsd, markers)
    im_wsd = (im_wsd == -1)
    im_wsd = np.asarray(im_wsd, np.uint8)
    return im_wsd
