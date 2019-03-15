'''
Loss functions
'''

import numpy as np

def categorical_crossentropy(y_true, y_pred):
	return -np.sum(y_true * np.log(y_pred))

def d_categorical_crossentropy_softmax(y_true, y_pred):
	x = y_pred[:,0]
	# initialize the 2-D jacobian matrix.
	jacobian_m = np.diag(x)

	for i in range(len(jacobian_m)):
		for j in range(len(jacobian_m)):
			if i == j:
				jacobian_m[i][j] = (x[i]-1)
			else:
				jacobian_m[i][j] = x[j]

	return y_true.T.dot(jacobian_m).T