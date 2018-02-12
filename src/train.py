from chainer.optimizers import Adam
from imclass import ImClass
from epoch import training_epoch, testing_epoch
from FCN_Classifier import FCN

import pickle


# training step
def train_step(fname_train="", fname_label="", fname_model="",
               N_train=25000, N_test=3000, N_epoch=1, batchsize=100, hgh=32, wid=32):

    print("Start training.")
    print("N_train:", N_train, "N_test:", N_test, "hgh:", hgh, "wid:", wid)

    cim = ImClass('train', fname_train=fname_train, fname_label=fname_label,
                  N_train=N_train, N_test=N_test, hgh=hgh, wid=wid)

    model = FCN()
    optimizer = Adam()
    optimizer.setup(model)

    # Learning loop
    for epoch in range(1, N_epoch + 1):
        print("epoch:", epoch, "/", N_epoch)

        # training
        sum_loss_training, sum_acc_training = 0.0, 0.0
        for i in range(0, N_train, batchsize):
            train_loss_tmp, train_acc_tmp = training_epoch(
                i, cim, model, optimizer, batchsize)

            sum_loss_training += float(train_loss_tmp) * batchsize
            sum_acc_training += float(train_acc_tmp) * batchsize

            if i == 0 or (i + batchsize) % 5000 == 0:
                print("training:", i + batchsize, "/", N_train,
                      "loss:", "{:.3f}".format(float(train_loss_tmp)),
                      "acc:", "{:.3f}".format(float(train_acc_tmp)))

        train_loss, train_acc = sum_loss_training / N_train, sum_acc_training / N_train

    # testing
    sum_acc_testing = 0.0
    for i in range(0, N_test, batchsize):
        test_acc_tmp = testing_epoch(i, cim, model, batchsize)
        sum_acc_testing += float(test_acc_tmp) * batchsize

        if (i + batchsize) % 1000 == 0:
            print("testing:", i + batchsize, "/", N_test,
                  "acc:", "{:.3f}".format(float(test_acc_tmp)))

    test_acc = sum_acc_testing / N_test

    print("Result", "\n",
          "train_loss:", "{:.3f}".format(train_loss), "\n",
          "train_acc:", "{:.3f}".format(train_acc), "\n",
          "test_acc:", "{:.3f}".format(test_acc))

    data_model = {}
    data_model['model'] = model
    data_model['shape'] = (hgh, wid)
    data_model['testacc'] = test_acc
    with open(fname_model, 'wb') as p:
        pickle.dump(data_model, p, -1)

    print("Done training.")


if __name__ == '__main__':
    # filenames for train
    fname_train = "../data/embryos_train.tiff"  # image
    fname_label = "../data/embryos_label.tiff"  # label
    fname_model = "../data/model.pkl"

    # parameters for train
    N_test = 3000
    N_train = 25000
    N_epoch = 1
    batchsize = 100
    hgh = 32
    wid = 32

    train_step(fname_train=fname_train, fname_label=fname_label, fname_model=fname_model,
               N_test=N_test, N_train=N_train, N_epoch=N_epoch, batchsize=batchsize, hgh=hgh, wid=wid)
