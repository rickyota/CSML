import numpy as np
from epoch import infer_epoch
from processing import clean_im, postprocessing


# infer a whole image
def infer_im(model, shape, im, thre_discard, wid_dilate):
    im_raw = infer_im_raw(model, shape, im)

    im_post = postprocessing(im_raw, thre_discard, wid_dilate)
    im_post = clean_im(im_post)
    im_post = np.asarray(im_post, np.uint8)

    return im_post


# get image after adapting max of two
def infer_im_raw(model, shape, im):
    shape_ori = im.shape
    if im.shape[0] % shape[0] != 0:
        im = np.lib.pad(
            im, ((0, shape[0] - im.shape[0] % shape[0]), (0, 0)), 'edge')
    if im.shape[1] % shape[1] != 0:
        im = np.lib.pad(
            im, ((0, 0), (0, shape[1] - im.shape[1] % shape[1])), 'edge')

    iml = combine_im(model, shape, im, 'large')
    ims = combine_im(model, shape, im, 'small')

    im = np.maximum(iml, ims)
    im = im[0:shape_ori[0], 0:shape_ori[1]]

    return im


# get image directly output from classifier
def combine_im(model, shape, im, imtype):
    hgh = shape[0]
    wid = shape[1]
    shapex = (1,) + shape

    if imtype == 'large':
        d = 0
    elif imtype == 'small':
        d = 1

    im_inferred = np.zeros_like(im)
    for i in range(int((d / 2) * hgh), im.shape[0] - hgh + 1, hgh):
        for j in range(int((d / 2) * wid), im.shape[1] - wid + 1, wid):
            patch = im[i:i + hgh, j:j + wid]
            patch = np.asarray(patch, np.float32).reshape((-1,) + shapex)
            patch = infer_epoch(model, patch)

            im_inferred[i + 1:i + hgh - 1, j + 1:j + wid - 1] = \
                patch[0, 1:-1, 1:-1]

    return im_inferred
