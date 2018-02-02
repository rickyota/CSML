import numpy as np
from cv2 import connectedComponents, connectedComponentsWithStats, watershed, dilate, distanceTransform, DIST_L2
from epoch import infer_epoch


# infer whole images
def infer_imwhole(model, cim, thre_discard, wid_dilate, thre_fill):
    """
    if cim.type_infer == 'file':
            numframe = cim.get_numframe()

            im_infer = []
            for i in range(numframe):
                    x_whole_each = cim.load_xwhole(i)
                    im_infer_each = infer_imwhole_each(model, cim, x_whole_each, thre_discard, wid_dilate, thre_fill)
                    im_infer.append(im_infer_each)
                    if i % 10 == 9:
                            print("Done inferring", i + 1, "/", numframe)

    elif cim.type_infer == 'folder':
            fnames = cim.fnames_infer

            im_infer = []
            for i, filename in enumerate(fnames):
                    x_whole_each = cim.load_one_image(filename)
                    im_infer_each = infer_imwhole_each(model, cim, x_whole_each, thre_discard, wid_dilate, thre_fill)
                    im_infer.append(im_infer_each)
                    if i % 10 == 9:
                            print("Done inferring", i + 1, "/", len(fnames))
    """

    num_im = cim.get_numframe() if cim.type_infer == 'file' else len(cim.fnames_infer)
    im_infer = []
    for i in range(num_im):
        if cim.type_infer == 'file':
            x_whole_each = cim.load_xwhole(i)
        else:
            x_whole_each = cim.load_one_image(cim.fnames_infer[i])
        im_infer_each = infer_imwhole_each(
            model, cim, x_whole_each, thre_discard, wid_dilate, thre_fill)
        im_infer.append(im_infer_each)
        if i % 5 == 4:
            print("Done inferring", i + 1, "/", num_im)

    im_infer = clean_ims(im_infer)
    im_infer = [np.asarray(im, np.uint8) for im in im_infer]

    return im_infer

# last operation for image saving


def clean_ims(im_infer):
    def clean_im(im):
        im = im * 255
        im[im != 255] = 0
        return im
    im_infer = list(map(clean_im, im_infer))
    return im_infer


# infer a whole image
def infer_imwhole_each(model, cim, x_whole, thre_discard, wid_dilate, thre_fill):
    im_infer_each = combine_im(model, cim, x_whole)

    im_infer_each = postprocessing(
        im_infer_each, thre_discard, wid_dilate, thre_fill)

    im_infer_each = np.asarray(im_infer_each, np.uint8)

    return im_infer_each


# get image after adapting max of two
def combine_im(model, cim, x_whole):
    shape_ori = x_whole.shape
    if x_whole.shape[0] % cim.hgh != 0:
        x_whole = np.lib.pad(
            x_whole, ((0, cim.hgh - x_whole.shape[0] % cim.hgh), (0, 0)), 'edge')
    if x_whole.shape[1] % cim.wid != 0:
        x_whole = np.lib.pad(
            x_whole, ((0, 0), (0, cim.wid - x_whole.shape[1] % cim.wid)), 'edge')

    iml = each_im(model, cim, x_whole, 'l')
    ims = each_im(model, cim, x_whole, 's')

    im_infer = np.maximum(iml, ims)
    im_infer = im_infer[0:shape_ori[0], 0:shape_ori[1]]

    return im_infer


# get image directly output from classifier
def each_im(model, cim, x_whole, imtype):
    if imtype == 'l':
        d = 0
    elif imtype == 's':
        d = 1

    x_batch = []
    x_result = []
    k = 0
    for i in range(int((d / 2) * cim.hgh), x_whole.shape[0] - cim.hgh + 1, cim.hgh):
        for j in range(int((d / 2) * cim.wid), x_whole.shape[1] - cim.wid + 1, cim.wid):
            x_batch.append(x_whole[i:i + cim.hgh, j:j + cim.wid])

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

    im_result = np.zeros_like(x_whole)
    k = 0
    for i in range(int((d / 2) * cim.hgh), x_whole.shape[0] - cim.hgh + 1, cim.hgh):
        for j in range(int((d / 2) * cim.wid), x_whole.shape[1] - cim.wid + 1, cim.wid):
            im_result[i:i + cim.hgh, j:j + cim.wid] = x_result[k]
            k += 1

    return im_result


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
    n, im_label, stat, _ = connectedComponentsWithStats(im, connectivity=4)
    im_discard = np.zeros_like(im)
    k = 0
    for i in range(1, n):
        if stat[i][4] > threshold:
            k += 1
            im_discard[im_label == i] = 1
    return im_discard


# dilate width px
def dilate_im(im, width):
    neibor8 = np.ones((width + 2, width + 2), np.int8)
    im = dilate(im, neibor8, iterations=1)
    return im


# fill area under threshld
def fill(im, threshold):
    im = 1 - im
    n, im_label, stat, _ = connectedComponentsWithStats(im, connectivity=4)
    for i in range(1, n):
        if stat[i][4] <= threshold:
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
