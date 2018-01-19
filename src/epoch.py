import numpy as np
from chainer import Variable


# train model in each epoch
def train_epoch(model, optimizer, x_train, t_train, batchsize=100):
	print("train epoch")
	N_train = t_train.shape[0]
	perm = np.random.permutation(N_train)
	sum_loss, sum_acc = 0, 0

	for i in range(0, N_train, batchsize):
		x_batch, t_batch = x_train[perm[i:i + batchsize]], t_train[perm[i:i + batchsize]]
		x_batch, t_batch = Variable(x_batch), Variable(t_batch)   

		model.zerograds()
		loss, acc = model(x_batch, t_batch)
		loss.backward()
		optimizer.update()

		sum_loss += float(loss.data) * batchsize
		sum_acc += float(acc.data) * batchsize
		if i % 10000 == 0:
			print("training:", i, "loss:", "{:.3f}".format(float(loss.data)), "acc:", "{:.3f}".format(float(acc.data)))

	train_loss_t, train_acc_t = sum_loss / N_train, sum_acc / N_train

	return train_loss_t, train_acc_t


# test accuracy in each epoch
def test_epoch(model, x_test, t_test, batchsize=100):
	print("test epoch")
	N_test = t_test.shape[0]
	sum_acc = 0

	for i in range(0, N_test, batchsize):
		x_batch, t_batch = x_test[i:i + batchsize], t_test[i:i + batchsize]
		x_batch, t_batch = Variable(x_batch), Variable(t_batch)
		_, acc = model(x_batch, t_batch)
		sum_acc += float(acc.data) * batchsize

		if i % 1000 == 0:
			print("testing:", i, "acc:", "{:.3f}".format(float(acc.data)))

	test_acc_t = sum_acc / N_test

	return test_acc_t


# infer images through classifiers
def infer_epoch(model, x_batch):
	x_batch_t = Variable(x_batch)
	x_infer_t = model.forward(x_batch_t)
	x_infer_t = np.argmax(x_infer_t.data, axis=1)
	x_infer = x_infer_t.astype(np.int)

	return x_infer

