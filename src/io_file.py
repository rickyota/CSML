import os
import numpy as np
import warnings
from PIL import Image
import pandas as pd


def load_ims_train(dname):
    if os.path.isdir(dname):
        fnames = get_listdir(dname)
        if not fnames:
            raise FileNotFoundError(
                "Files not found in the folder: {}.".format(dname))

        ims = []
        for fname in fnames:
            im = read_im_any(fname)
            ims.extend(im)
    else:
        raise FileNotFoundError(
            "Folder not found: {}.".format(dname))

    ims = np.asarray(ims, np.float32)

    # if there is only one image
    if len(ims.shape) == 2:
        ims = ims.reshape((-1, ims.shape[0], ims.shape[1]))
    return ims


def load_ims_label(dname):
    if os.path.isdir(dname):
        fnames = get_listdir(dname)
        if not fnames:
            raise FileNotFoundError(
                "No files in the folder: {}.".format(dname))
        ims = []
        for filename in fnames:
            im = read_im_any(filename)
            ims.extend(im)
    else:
        raise FileNotFoundError(
            "No file or folder found: {}.".format(dname))

    ims = np.asarray(ims, np.int32)

    # if there is only one image
    if len(ims.shape) == 2:
        ims = ims.reshape((-1, ims.shape[0], ims.shape[1]))
    return ims


def scaling(im):
    return (im - im.min()) / (im.max() - im.min())


def read_im_any(fname, scale=False):
    if fname.endswith('.tif') or fname.endswith('.tiff'):
        return read_im_tif(fname, scale)
    else:
        return [read_im(fname, scale)]


def read_im(fname, scale=False):
    try:
        fim = Image.open(fname)
    except Exception as e:
        raise IOError(e, "Cannot open file: {}".format(fname))

    im = np.asarray(fim.convert('L')) / 255.0
    if scale:
        im = scaling(im)
    return im


def read_im_tif(fname, scale):
    fim, num = read_fim(fname)

    ims = []
    for i in range(num):
        fim.seek(i)
        im_tmp = np.asarray(fim.convert('L')) / 255.0
        if scale:
            im_tmp = scaling(im_tmp)
        ims.append(im_tmp)
    return ims


def read_fim(fname):
    try:
        fim = Image.open(fname)
    except Exception as e:
        raise IOError(e, "Cannot open file: {}".format(fname))

    num = 0

    try:
        while True:
            fim.seek(num)
            num = num + 1
    except EOFError:
        pass

    return fim, num


def get_listdir(dname):
    """get filename in dir
    """
    fnames = os.listdir(dname)
    fnames = list(filter(lambda f: not f.startswith("."), fnames))
    fnames.sort()
    fnames = [os.path.join(dname, fname) for fname in fnames]
    return fnames


# make inferred filename
def get_listdir_inferred(dname_inferred, dname_infer):
    """get filename in dir
    """
    fnames = os.listdir(dname_infer)
    fnames = list(filter(lambda f: not f.startswith("."), fnames))
    fnames.sort()
    fnames = [os.path.join(dname_inferred, fname) for fname in fnames]
    fnames = [os.path.splitext(fname)[0] + '.png'
              if not(fname.endswith('.tif') or fname.endswith('.tiff')) else fname
              for fname in fnames]
    return fnames


def get_listdir_inferred_tag(fname_tag, fnames_inferred):
    fnames = [os.path.join(fname_tag, os.path.basename(fname))
              for fname in fnames_inferred]
    return fnames


def get_listdir_inferred_stats(fname_stats, fnames_inferred):
    fnames = [os.path.join(fname_stats, os.path.splitext(os.path.basename(fname))[0]) + '.xlsx'
              for fname in fnames_inferred]
    return fnames


def save_im(im, fname, mode=None):
    """save image
    """
    if mode is not None:
        fim = Image.fromarray(im, mode=mode)
    else:
        fim = Image.fromarray(im)
    try:
        if os.path.isfile(fname):
            warnings.warn(
                "File is being overwritten: {}.".format(fname))
        fim.save(fname)
    except Exception as e:
        raise Exception(
            e, "Cannot save image to file: {}".format(fname))


def save_im_tif(ims, fname, mode=None):
    """save images to one tif file
    """
    if mode is not None:
        ims = [Image.fromarray(im, mode=mode) for im in ims]
    else:
        ims = [Image.fromarray(im) for im in ims]
    try:
        if os.path.isfile(fname):
            warnings.warn(
                "File is being overwritten: {}.".format(fname))
        if len(ims) == 1:
            ims[0].save(fname, save_all=True)
        else:
            ims[0].save(fname, save_all=True, append_images=ims[1:])
    except Exception as e:
        raise Exception(
            e, "Cannot save images into file: {}".format(fname))


def save_xlsx(dfs, fname):
    if os.path.isfile(fname):
        warnings.warn(
            "File is being overwritten: {}.".format(fname))
    if isinstance(dfs, list):
        writer = pd.ExcelWriter(fname)
        for i, df in enumerate(dfs):
            df.to_excel(writer, 'Image' + str(i + 1))
    else:
        dfs.to_excel(fname, 'Image1')
