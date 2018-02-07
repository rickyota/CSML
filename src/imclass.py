import numpy as np
from cv2 import connectedComponents
from PIL import Image
import os
from random import shuffle
import warnings


class ImClass:

    def __init__(self, usetype,  fname_train="", fname_label="",
                 fname_infer="", fname_inferred="",
                 N_train=25000, N_test=3000, hgh=32, wid=32):

        # input size of classifier
        self.hgh = hgh
        self.wid = wid
        self.shapex = [1, self.hgh, self.wid]
        self.shapet = [self.hgh, self.wid]

        if usetype == 'train':
            self.__ims_train = self._load_ims_train(fname_train)
            self.__ims_label = self._load_ims_label(fname_label)

            if self.__ims_train.shape[0] != self.__ims_label.shape[0]:
                raise ValueError(
                    "Number of images in two folders or files are different: \
                     {0} and {1}".format(fname_train, fname_label))
            if self.__ims_train.shape[1] != self.__ims_label.shape[1] or \
                    self.__ims_train.shape[2] != self.__ims_label.shape[2]:
                raise ValueError(
                    "Size of training images and label images are not same.")

            self.__poss_training, self.__poss_testing = \
                self._make_dataset(
                    self.__ims_label, N_train, N_test)

        elif usetype == 'infer':
            if os.path.isfile(fname_infer):
                self.type_infer = 'file'
                self.file_infer, self.num_infer \
                    = self._load_file(fname_infer)
                self.fname_inferred = fname_inferred

            elif os.path.isdir(fname_infer):
                self.type_infer = 'folder'

                if not os.path.isdir(fname_inferred):
                    os.mkdir(fname_inferred)

                self.fnames_infer = self._get_listdir(fname_infer)
                self.fnames_inferred = self._get_listdir_inferred(
                    fname_inferred, fname_infer)

            else:
                raise FileNotFoundError(
                    "No file or folder found: {}.".format(fname_infer))

    # use hgh and wid in model.pkl
    def change_hgh_wid(self, shape):
        self.hgh, self.wid = shape[0], shape[1]
        self.shapex = [1, self.hgh, self.wid]
        self.shapet = [self.hgh, self.wid]

    # load images
    def _load_ims_train(self, fname):
        if os.path.isfile(fname):
            fim, num = self._load_file(fname)
            ims = []
            for i in range(num):
                fim.seek(i)
                im_tmp = np.asarray(fim.convert('L')) / 255.0
                ims.append(im_tmp)

        elif os.path.isdir(fname):
            fnames = self._get_listdir(fname)
            if not fnames:
                raise FileNotFoundError(
                    "No files in the folder: {}.".format(fname))

            ims = []
            for filename in fnames:
                im_tmp = self.read_im_folder(filename)
                ims.append(im_tmp)
        else:
            raise FileNotFoundError(
                "No file or folder not found: {}.".format(fname))

        ims = np.asarray(ims, np.float32)

        # if there is only one image
        if len(ims.shape) == 2:
            ims = ims.reshape((-1, ims.shape[0], ims.shape[1]))

        return ims

    # load contoured images
    def _load_ims_label(self, fname):
        if os.path.isfile(fname):
            fim, num = self._load_file(fname)
            ims = []
            for i in range(num):
                fim.seek(i)
                im_tmp = np.asarray(fim.convert('L')) / 255
                ims.append(im_tmp)

        elif os.path.isdir(fname):
            fnames = self._get_listdir(fname)
            if not fnames:
                raise FileNotFoundError(
                    "No files in the folder: {}.".format(fname))
            ims = []
            for filename in fnames:
                im_tmp = self.read_im_folder(filename)
                ims.append(im_tmp)

        else:
            raise FileNotFoundError(
                "No file or folder found: {}.".format(fname))

        ims = np.asarray(ims, np.int32)

        if len(ims.shape) == 2:
            ims = ims.reshape((-1, ims.shape[0], ims.shape[1]))

        return ims

    # choose images suitable for small training images
    def _make_dataset(self, ims_label, N_train, N_test):
        N_each = int((N_train + N_test) / ims_label.shape[0])

        poss = []

        for i, im_label in enumerate(ims_label):
            bound = self._getBound(im_label)
            xs, ys = np.where(bound == True)

            if len(xs) < N_each:
                raise ValueError(
                    "Training image set cannot be made. Decrease 'number train' or \
                    spread contoured region in label images.")

            perm = np.random.permutation(xs.shape[0])
            xs, ys = xs[perm[0:2 * N_each]], ys[perm[0:2 * N_each]]

            count = 0
            for j in range(xs.shape[0]):
                x, y = xs[j], ys[j]
                if self._isInBound(bound, x, y):
                    pos = (i, (x, y))
                    poss.append(pos)
                    count += 1
                    if count >= N_each:
                        break

        shuffle(poss)

        poss_training = poss[0:N_train]
        poss_testing = poss[N_train:N_train + N_test]

        return poss_training, poss_testing

    # get criteria of whether suitable or not
    def _getBound(self, im_label):
        _, bound = connectedComponents(np.uint8(1 - im_label))
        bound[bound != 1] = 0
        bound = 1 - bound
        bound = np.asarray(bound, bool)
        return bound

    # judge based on criteria
    def _isInBound(self, bound, x, y):
        return x + self.hgh < bound.shape[0] and y + self.wid < bound.shape[1] \
            and bound[x + self.hgh, y] and bound[x, y + self.wid] \
            and bound[x + self.hgh, y + self.wid]

    def get_x_training_batch(self, i, batchsize):
        poss = self.__poss_training[i:i + batchsize]
        x_batch = self._make_x_batch(self.__ims_train, poss)
        return x_batch

    def get_t_training_batch(self, i, batchsize):
        poss = self.__poss_training[i:i + batchsize]
        t_batch = self._make_t_batch(self.__ims_label, poss)
        return t_batch

    def get_x_testing_batch(self, i, batchsize):
        poss = self.__poss_testing[i:i + batchsize]
        x_batch = self._make_x_batch(self.__ims_train, poss)
        return x_batch

    def get_t_testing_batch(self, i, batchsize):
        poss = self.__poss_testing[i:i + batchsize]
        t_batch = self._make_t_batch(self.__ims_label, poss)
        return t_batch

    def _make_x_batch(self, ims, poss):
        batch = []
        for pos in poss:
            im = ims[pos[0]]
            (x, y) = pos[1]
            im_cut = im[x:x + self.hgh, y:y + self.wid]
            batch.append(im_cut)
        batch = np.asarray(batch, np.float32)
        batch = batch.reshape([-1] + self.shapex)
        return batch

    def _make_t_batch(self, ims, poss):
        batch = []
        for pos in poss:
            im = ims[pos[0]]
            (x, y) = pos[1]
            im_cut = im[x:x + self.hgh, y:y + self.wid]
            batch.append(im_cut)
        batch = np.asarray(batch, np.int32)
        batch = batch.reshape([-1] + self.shapet)
        return batch

    # load num-th whole image
    def read_im_file(self, num):
        im = self.file_infer
        im.seek(num)
        im_xwhole = np.asarray(im.convert('L'), np.float32) / 255.0
        return im_xwhole

    # load image from non-tiff file
    def read_im_folder(self, fname):
        try:
            fim = Image.open(fname)
        except Exception as e:
            raise IOError(e, "Cannot open file: {}".format(fname))

        im = np.asarray(fim.convert('L')) / 255.0
        return im

    # load file and number of frame
    def _load_file(self, fname):
        try:
            fim = Image.open(fname)
        except Exception as e:
            raise IOError(e, "Cannot open file: {}".format(fname))

        num = 0

        try:
            while True:
                fim.seek(num)
                num += 1
        except EOFError:
            pass

        return fim, num

    # git filename in dir
    def _get_listdir(self, fname):
        fnames = os.listdir(fname)
        fnames = list(filter(lambda f: not f.startswith("."), fnames))
        fnames = [fname + "/" + filename for filename in fnames]
        return fnames

    # make inferred filename
    def _get_listdir_inferred(self, fname_inferred, fname_infer):
        fnames = os.listdir(fname_infer)
        fnames = list(filter(lambda f: not f.startswith("."), fnames))
        fnames = [fname_inferred + "/" + filename for filename in fnames]
        fnames = [os.path.splitext(filename)[0] + '.png'
                  for filename in fnames]
        return fnames

    # save image
    def save_im(self, im, fname):
        # folder
        if self.type_infer == 'folder':
            im = Image.fromarray(im)
            try:
                if os.path.isfile(fname):
                    warnings.warn(
                        "File is being overwritten: {}.".format(fname))
                im.save(fname)
            except Exception as e:
                raise Exception(
                    e, "Cannot save image to file: {}".format(fname))

        # file
        elif self.type_infer == 'file':
            im = [Image.fromarray(img) for img in im]
            try:
                if os.path.isfile(fname):
                    warnings.warn(
                        "File is being overwritten: {}.".format(fname))
                if len(im) == 1:
                    im[0].save(fname, save_all=True)
                else:
                    im[0].save(fname, save_all=True, append_images=im[1:])
            except Exception as e:
                raise Exception(
                    e, "Cannot save images into file: {}".format(fname))
