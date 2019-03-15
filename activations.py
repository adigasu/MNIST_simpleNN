'''
Activation functions
'''

import numpy as np

def relu(x):
	x[x<=0] = 0
	return x

def softmax(x):
    out = np.exp(x)
    return out/np.sum(out)

def softmax_stable(x):
    out = np.exp(x - np.max(x))
    return out/np.sum(out)

def d_softmax(x):
	x = x[:,0]
	# initialize the 2-D jacobian matrix.
	jacobian_m = np.diag(x)

	for i in range(len(jacobian_m)):
		for j in range(len(jacobian_m)):
			if i == j:
				jacobian_m[i][j] = x[i] * (1-x[i])
			else:
				jacobian_m[i][j] = - x[i] * x[j]

	return jacobian_m


def sigmoid(x):
	out = np.exp(-x)
	return 1./(1+out)

def d_sigmoid(x):
	return x * (1-x)
