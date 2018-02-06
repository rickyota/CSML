from chainer.optimizers import Adam
from imclass import ImClass
from epoch import training_epoch, testing_epoch
from FCN_Classifier import FCN

import pickle


# training step
def train_step(fname_train="", fname_label="", fname_model="",
               N_train=25000, N_test=3000, N_epoch=1, batchsize=100, hgh=32, wid=32):

    print("start training.")

    cim = ImClass('train', fname_train=fname_train, fname_label=fname_label,
                  N_train=N_train, N_test=N_test, hgh=hgh, wid=wid)
    train_training, label_training, train_testing, label_testing = cim.load_batch()
    print("shapes:", train_training.shape, label_training.shape,
          train_testing.shape, label_testing.shape)

    model = FCN()
    optimizer = Adam()
    optimizer.setup(model)

    train_loss, train_acc, test_acc = [], [], []

    # Learning loop
    for epoch in range(1, N_epoch + 1):
        print("epoch: ", epoch, "/", N_epoch)
        # training
        (train_loss_tmp, train_acc_tmp) = training_epoch(
            model, optimizer, train_training, label_training, batchsize)
        train_loss.append(train_loss_tmp), train_acc.append(train_acc_tmp)

        # evaluation
        test_acc_tmp = testing_epoch(
            model, train_testing, label_testing, batchsize)
        test_acc.append(test_acc_tmp)

        print("train_loss", "train_acc", "test_acc", "\n",
              "{:.3f}".format(train_loss_tmp), "{:.3f}".format(train_acc_tmp),
              "{:.3f}".format(test_acc_tmp))

    data_model = {}
    data_model['model'] = model
    data_model['shape'] = (hgh, wid)
    data_model['testacc'] = test_acc
    with open(fname_model, 'wb') as p:
        pickle.dump(data_model, p, -1)

    print("done training.")


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
