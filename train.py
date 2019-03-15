'''
Train simaple NN
'''

import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from utils import *
from activations import *
from losses import *

############ functions ###########

def forward_backward_pass(img, label, weights):

	# Forward prop
	[w1, w2, w3, b1, b2, b3] = weights

	fc1 = w1.dot(img) + b1
	fc1 = relu(fc1) 	# Relu activation

	fc2 = w2.dot(fc1) + b2
	fc2 = relu(fc2) 	# Relu activation

	fc3 = w3.dot(fc2) + b3
	pred = softmax_stable(fc3) 	# Softmax activation to get final output

	# categorical cross-entropy loss function
	loss = categorical_crossentropy(label, pred)

	# Backward prop using difference loss
	# dpred = pred - label
	# dpred = dpred.T.dot(d_softmax(pred)).T

	# Backward prop using categorical_crossentropy
	dpred = d_categorical_crossentropy_softmax(label, pred)
	dw3 = dpred.dot(fc2.T)
	db3 = np.sum(dpred, axis=1).reshape(b3.shape)

	dtemp = w3.T.dot(dpred)
	dtemp = dtemp*relu(fc2)
	dw2 = dtemp.dot(fc1.T)
	db2 = np.sum(dtemp, axis=1).reshape(b2.shape)

	dtemp = w2.T.dot(dtemp)
	dtemp = dtemp*relu(fc1)
	dw1 = dtemp.dot(img.T)
	db1 = np.sum(dtemp, axis=1).reshape(b1.shape)

	gradients = [dw1, dw2, dw3, db1, db2, db3]

	return gradients, loss

def SGD(batch, n_classes, lr, img_dim, weights, loss):
	X_train = batch[:,:-1]
	Y_train = batch[:,-1]

	loss_b = 0
	batch_size = batch.shape[0]
	[w1, w2, w3, b1, b2, b3] = weights

	dw1 = np.zeros_like(w1)
	dw2 = np.zeros_like(w2)
	dw3 = np.zeros_like(w3)
	db1 = np.zeros_like(b1)
	db2 = np.zeros_like(b2)
	db3 = np.zeros_like(b3)

	v1 = np.zeros_like(w1)
	v2 = np.zeros_like(w2)
	v3 = np.zeros_like(w3)
	bv1 = np.zeros_like(b1)
	bv2 = np.zeros_like(b2)
	bv3 = np.zeros_like(b3)

	for i in range(batch_size):
		x = X_train[i].reshape(X_train.shape[-1], 1)
		y = np.eye(n_classes)[int(Y_train[i])]
		y = y.reshape(n_classes, 1)

		# Find gradients
		grads, loss_ = forward_backward_pass(x, y, weights)
		[dw1_, dw2_, dw3_, db1_, db2_, db3_] = grads

		dw1 += dw1_
		dw2 += dw2_
		dw3 += dw3_
		db1 += db1_
		db2 += db2_
		db3 += db3_

		loss_b += loss_

	# Update SGD gradients (TODO: add momentum terms)
	w1 -= lr * dw1/batch_size
	w2 -= lr * dw2/batch_size
	w3 -= lr * dw3/batch_size
	b1 -= lr * db1/batch_size
	b2 -= lr * db2/batch_size
	b3 -= lr * db3/batch_size

	loss_b = loss_b/batch_size
	loss.append(loss_b)

	weights = [w1, w2, w3, b1, b2, b3]

	return weights, loss

def update_lr(lr, itr, decay=1e-6):
	lr *= (1./(1.+decay*itr))		# Exponential decay of learning rate
	return lr

######### test function #############
def forward_pass_predict(img, label, weights):
	# Forward prop
	[w1, w2, w3, b1, b2, b3] = weights

	fc1 = w1.dot(img) + b1
	fc1 = relu(fc1) 	# Relu activation

	fc2 = w2.dot(fc1) + b2
	fc2 = relu(fc2) 	# Relu activation

	fc3 = w3.dot(fc2) + b3
	pred = softmax_stable(fc3) 	# Softmax activation to get final output

	# categorical cross-entropy loss function
	loss = categorical_crossentropy(label, pred)

	# Prediction
	y_pred = (pred == np.max(pred))
	y_pred = y_pred*1.
	y_pred = sum(y_pred==label)/10.		# For correct prediction y_pred : 1.0 else 0.8 (wrong prediction)

	return loss, y_pred

def predict(batch, n_classes, weights, y_pred, loss):
	X_train = batch[:,:-1]
	Y_train = batch[:,-1]
	batch_size = batch.shape[0]
	loss_b = 0

	# Predict test images
	for i in range(batch_size):
		x = X_train[i].reshape(X_train.shape[-1], 1)
		y = np.eye(n_classes)[int(Y_train[i])]
		y = y.reshape(n_classes, 1)
		loss_, pred_ = forward_pass_predict(x, y, weights)
		loss_b += loss_
		y_pred.append(pred_)

	loss_b = loss_b/batch_size
	loss.append(loss_b)

	return y_pred, loss


########## Main function ###########

if __name__ == '__main__':

	# Parameters
	n_classes = 10					# Number of classes
	img_H = 28						# Image width
	img_W = 28						# Image height
	img_ch = 1						# No. of channels
	img_dim = (img_H, img_W, 1)		# Image dimension
	batch_size = 32					# Batch size
	n_epochs = 30					# No. of epochs
	lr = 0.1						# Initial learning rate
	decay = 2e-8					# Learning rate decay

	# Read data Train data
	n_img = 50000
	X_train = read_data('data/train-images-idx3-ubyte.gz', n_img, [img_H, img_W, img_ch])
	Y_train = read_label('data/train-labels-idx1-ubyte.gz', n_img)

	# Normalize data
	X_train /=np.max(X_train)
	X_train = X_train.reshape(n_img, img_H * img_W * img_ch)
	train_data = np.hstack((X_train, Y_train))

	# Read data test data & Normalize it
	t_img = 10000
	X_test = read_data('data/t10k-images-idx3-ubyte.gz', t_img, [img_H, img_W, img_ch])
	Y_test = read_label('data/t10k-labels-idx1-ubyte.gz', t_img)
	X_test /=np.max(X_test)
	X_test = X_test.reshape(t_img, img_H * img_W * img_ch)
	test_data = np.hstack((X_test, Y_test))

	# Initialize network weights and bias, Architecture: [input, 256, 64, 10]
	w1 = init_fc_weight((256, X_train.shape[1]))
	w2 = init_fc_weight((64, 256))
	w3 = init_fc_weight((n_classes, 64))
	b1 = np.zeros((w1.shape[0],1))
	b2 = np.zeros((w2.shape[0],1))
	b3 = np.zeros((w3.shape[0],1))

	weights = [w1, w2, w3, b1, b2, b3]
	loss = []							# save loss for every iterations
	itr = 1								# Counter for iteration
	test_flag =1						# Test flag
	b_acc = 0.0							# best accuracy
	save_path = './weights/'			# Save weights path
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	# Training starts here
	for epoch in range(n_epochs):
		np.random.shuffle(train_data)
		batches = [train_data[k:k + batch_size] for k in range(0, train_data.shape[0], batch_size)]

		t = tqdm(batches)
		for x, batch in enumerate(t):
			weights, loss = SGD(batch, n_classes, lr, img_dim, weights, loss)
			t.set_description("Loss: %.4f, epochs: %d, lr = %.4f" % (loss[-1], epoch, lr))
			lr = update_lr(lr, itr, decay)
			itr += 1

		# Test after each epoch
		if (test_flag ==1):
			y_pred = []
			t_loss = []
			np.random.shuffle(test_data)
			batches = [test_data[k:k + batch_size] for k in range(0, test_data.shape[0], batch_size)]

			t = tqdm(batches)
			for x, batch in enumerate(t):
				y_pred, t_loss = predict(batch, n_classes, weights, y_pred, t_loss)

			new_acc = sum(np.array(y_pred)==1)*100./t_img
			print('Accuracy = %.4f' %(new_acc))
			print('')

			# Save weights
			if new_acc > b_acc :
				b_acc = new_acc
				save_val = [weights, loss[-1], epoch, new_acc]
				save_file = save_path + str(epoch) + '_' + str(new_acc) + '_weights.pkl'
				with open(save_file, 'wb') as file:
					pickle.dump(save_val, file)


	# Plot loss values
	plt.plot(loss, 'b', label='Training loss')
	plt.xlabel('iterations')
	plt.ylabel('loss')
	plt.legend(loc='upper right')
	plt.show()
