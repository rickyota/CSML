import numpy as np
from cv2 import connectedComponents
import os
from random import shuffle
from io_file import load_ims_train, load_ims_label, read_im, read_im_tif, \
    save_im, save_im_tif, save_xlsx, \
    get_listdir, get_listdir_inferred, get_listdir_inferred_tag, get_listdir_inferred_stats


class ImClass:
    def __init__(self, usetype, fname_train="", fname_label="",
                 fname_infer="", fname_inferred="",
                 N_train="", N_test="", hgh=64, wid=64,
                 mode='back'):

        # input size of classifier
        self.hgh = hgh
        self.wid = wid
        self.shape = (self.hgh, self.wid)
        self.shapex = (1, self.hgh, self.wid)
        self.shapet = (self.hgh, self.wid)
        self.hhgh = int(hgh / 2)
        self.hwid = int(wid / 2)

        self.mode = mode

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
            if os.path.isdir(fname_infer):

                if not os.path.isdir(fname_inferred):
                    os.mkdir(fname_inferred)

                self.fnames_infer = get_listdir(fname_infer)
                self.fnames_inferred = get_listdir_inferred(
                    fname_inferred, fname_infer)
                self.fnames_tag = get_listdir_inferred_tag(
                    self.fnames_inferred)
                self.fnames_stats = get_listdir_inferred_stats(
                    self.fnames_inferred)

            else:
                raise FileNotFoundError(
                    "No file or folder found: {}.".format(fname_infer))

    # use hgh and wid in model.pkl
    def change_hgh_wid(self, shape):
        self.hgh, self.wid = shape[0], shape[1]
        self.shapex = (1, self.hgh, self.wid)
        self.shapet = (self.hgh, self.wid)
        self.shape = (self.hgh, self.wid)

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

    def read_im(self, fname, scale=False):
        return read_im(fname, scale=scale)

    def read_im_tif(self, fname, scale=False):
        return read_im_tif(fname, scale=scale)

    def save_im(self, im, fname):
        save_im(im, fname)

    def save_im_tif(self, ims, fname):
        save_im_tif(ims, fname)

    def save_im_tag(self, im, fname):
        save_im(im, fname, mode='I')

    def save_im_tif_tag(self, ims, fname):
        save_im_tif(ims, fname, mode='I')

    def save_xlsx(self, df, fname):
        save_xlsx(df, fname)

    # load images
    def _load_ims_train(self, fname):
        return load_ims_train(fname)

    # load contoured images
    def _load_ims_label(self, fname):
        return load_ims_label(fname)

    # choose images suitable for small training images
    def _make_dataset(self, ims_label, N_train, N_test):
        N_each = int((N_train + N_test) / ims_label.shape[0]) + 1

        if self.mode == 'whole':
            in_bound = self._isInBound
        elif self.mode == 'back' or self.mode == 'all':
            in_bound = self._isInBound_cent

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
                if in_bound(bound, x, y):
                    pos = (i, (x, y))
                    poss.append(pos)
                    count += 1
                    if count >= N_each:
                        break

        if len(poss) < N_train + N_test:
            raise ValueError(
                "Training image set cannot be made. Decrease 'number train' or \
                spread contoured region in label images.")

        shuffle(poss)

        poss_training = poss[0:N_train]
        poss_testing = poss[N_train:N_train + N_test]

        return poss_training, poss_testing

    # get criteria of whether suitable or not
    def _getBound(self, im_label):
        """mode: 
        'back','whole','all'
        """

        if self.mode == 'whole' or self.mode == 'back':
            _, bound = connectedComponents(
                np.uint8(1 - im_label))
            bound[bound != 1] = 0
            bound = 1 - bound
            bound = np.asarray(bound, bool)
            return bound

        elif self.mode == 'all':
            bound = np.ones_like(im_label)
            bound = np.asarray(bound, bool)
            return bound

        else:
            raise ValueError("'mode' not found")

    # judge based on criteria
    def _isInBound(self, bound, x, y):
        hhgh = self.hhgh
        hwid = self.hwid
        return 0 <= x - hhgh and x + hhgh < bound.shape[0] \
            and 0 <= y - hwid and y + hwid < bound.shape[1] \
            and bound[x - hhgh, y - hwid] and bound[x - hhgh, y + hwid] \
            and bound[x + hhgh, y - hwid] and bound[x + hhgh, y + hwid]

    def _isInBound_cent(self, bound, x, y):
        hhgh = self.hhgh
        hwid = self.hwid
        return 0 <= x - hhgh and x + hhgh < bound.shape[0] \
            and 0 <= y - hwid and y + hwid < bound.shape[1]

    def _patch(self, ims, pos):
        return ims[pos[0]][pos[1][0] - self.hhgh:pos[1][0] + self.hhgh,
                           pos[1][1] - self.hwid:pos[1][1] + self.hwid]

    def _make_x_batch(self, ims, poss):
        batch = [[self._patch(ims, pos)] for pos in poss]
        batch = np.array(batch, np.float32)
        batch = batch.reshape((-1,) + self.shapex)
        return batch

    def _make_t_batch(self, ims, poss):
        batch = [[self._patch(ims, pos)] for pos in poss]
        batch = np.array(batch, np.int32)
        batch = batch.reshape((-1,) + self.shapet)
        return batch
