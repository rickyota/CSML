from chainer import Chain
from chainer.initializers import HeNormal
from chainer import links as L
from chainer import functions as F


# class of FCN Classifier
class FCN(Chain):

    def __init__(self):
        initializer = HeNormal()
        self.n_layer1 = 16
        self.n_layer2 = 32
        super(FCN, self).__init__(
            conv1_1=L.Convolution2D(
                None, self.n_layer1, 3, stride=1, pad=1, initialW=initializer),
            conv1_2=L.Convolution2D(
                None, self.n_layer1, 3, stride=1, pad=1, initialW=initializer),
            conv2_1=L.Convolution2D(
                None, self.n_layer2, 3, stride=1, pad=1, initialW=initializer),
            conv2_2=L.Convolution2D(
                None, self.n_layer2, 3, stride=1, pad=1, initialW=initializer),
            convcha3=L.Convolution2D(
                None, 2, 1, stride=1, pad=0, initialW=initializer),
            deconvsize3=L.Deconvolution2D(
                None, 2, 4, stride=2, pad=1, initialW=initializer),
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

        h = F.relu(self.conv1_1(x_data))
        h = F.relu(self.conv1_2(h))
        h = self.norm1(h)
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = self.norm2(h)

        h = self.convcha3(h)
        h = self.deconvsize3(h)
        return h
