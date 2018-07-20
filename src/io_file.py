import os
import numpy as np
import warnings
from PIL import Image

print("a")


def load_ims_train(dname):
    if os.path.isdir(dname):
        fnames = get_listdir(dname)
        if not fnames:
            raise FileNotFoundError(
                "Files not found in the folder: {}.".format(dname))

        ims = []
        for fname in fnames:
            im = read_im(fname)
            ims.append(im)
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
            im = read_im(filename)
            ims.append(im)
    else:
        raise FileNotFoundError(
            "No file or folder found: {}.".format(dname))

    ims = np.asarray(ims, np.int32)

    # if there is only one image
    if len(ims.shape) == 2:
        ims = ims.reshape((-1, ims.shape[0], ims.shape[1]))
    return ims


def scaling(im):
    print("scaling", im.min(), im.max())
    return (im - im.min()) / (im.max() - im.min())


def read_im(fname, scale=False):
    try:
        fim = Image.open(fname)
    except Exception as e:
        raise IOError(e, "Cannot open file: {}".format(fname))

    im = np.asarray(fim.convert('L')) / 255.0
    if scale:
        im = scaling(im)
    return im


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
              for fname in fnames]
    return fnames


def save_im(im, fname):
    """save image
    """
    # folder
    fim = Image.fromarray(im)
    try:
        if os.path.isfile(fname):
            warnings.warn(
                "File is being overwritten: {}.".format(fname))
        fim.save(fname)
    except Exception as e:
        raise Exception(
            e, "Cannot save image to file: {}".format(fname))
