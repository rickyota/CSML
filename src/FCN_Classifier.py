import numpy as np
from chainer import Chain
from chainer.initializers import HeNormal
from chainer import links as L
from chainer import functions as F


# class of FCN Classifier
class FCN(Chain):

    def __init__(self):
        initializer = HeNormal()
        self.n_layer1 = 32
        self.n_layer2 = 64
        super(FCN, self).__init__(
            conv1_1=L.Convolution2D(
                1, self.n_layer1, 3, stride=1, pad=1, initialW=initializer),
            conv1_2=L.Convolution2D(
                self.n_layer1, self.n_layer1, 3, stride=1, pad=1, initialW=initializer),
            conv2_1=L.Convolution2D(
                self.n_layer1, self.n_layer2, 3, stride=1, pad=1, initialW=initializer),
            conv2_2=L.Convolution2D(
                self.n_layer2, self.n_layer2, 3, stride=1, pad=1, initialW=initializer),
            convcha3=L.Convolution2D(
                self.n_layer2, 2, 1, stride=1, pad=0, initialW=initializer),
            deconvsize3=L.Deconvolution2D(
                2, 2, 4, stride=2, pad=1, initialW=initializer),
            norm1=L.BatchNormalization(self.n_layer1),
            norm2=L.BatchNormalization(self.n_layer2),
        )

    # get loss function and accuracy
    def __call__(self, x_data, t_data):
        x_data_f = self.forward(x_data)
        loss = F.softmax_cross_entropy(x_data_f, t_data)
        acc = F.accuracy(x_data_f, t_data)
        return loss, acc

    # get images output from classifier
    def forward(self, x_data):
        h = self.im_enhance(x_data)
        h = h.astype(np.float32)

        h = F.relu(self.conv1_1(h))
        h = F.relu(self.conv1_2(h))
        h = self.norm1(h)
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = self.norm2(h)
        q3 = self.convcha3(h)
        r3 = self.deconvsize3(q3)
        return r3

    # enhance images before putting in layers
    # standardization -> sigmoid -> standardization
    def im_enhance(self, x_batch):
        try:
            x_batch_st = self.im_stan(x_batch.data)
            x_batch_st_med = np.max(x_batch_st, axis=(2, 3)) / 2
            x_batch_st_med = x_batch_st_med.reshape(
                (x_batch.data.shape[0], x_batch.data.shape[1], 1, 1))
            x_batch_sig = self.im_sigmoid(x_batch_st, x_batch_st_med, 1)
            x_batch_stan = self.im_stan(x_batch_sig)
            x_batch_stan = x_batch_stan.astype(np.float32)
        except Exception as e:
            print("Got exception.", e)
            x_batch_stan = x_batch.data.astype(np.float32)
        return x_batch_stan

    # standardize
    def im_stan(self, x_batch):
        x_batch_mean = x_batch.mean(axis=(2, 3))
        x_batch_std = x_batch.std(axis=(2, 3))
        if np.any(x_batch_std == 0):
            raise ZeroDivisionError("x_batch_std contains 0")
        x_batch_mean = x_batch_mean.reshape(
            (x_batch.shape[0], x_batch.shape[1], 1, 1))
        x_batch_std = x_batch_std.reshape(
            (x_batch.shape[0], x_batch.shape[1], 1, 1))
        x_batch_stan = (x_batch - x_batch_mean) / x_batch_std
        return x_batch_stan

    # adapt sigmoid function
    def im_sigmoid(self, x_batch, x_batch_med, a=1):
        x_batch_sig = 1.0 / (1.0 + np.exp(-a * (x_batch - x_batch_med)))
        return x_batch_sig
