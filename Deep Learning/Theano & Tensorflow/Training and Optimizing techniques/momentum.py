# Using MNIST data set-

import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

from utility import get_normalized_data, error_rate, cost, y2indicator
from mlp import forward, derivative_w2, derivative_b2, derivative_w1, derivative_b1

def main():
	# Compare 3 scenarios:
	# 1. Batch SGD
	# 2. Batch SGD with Regular Momentum
	# 3. Batch SGD with Nesterov Momentum

	max_iter = 20 # make it 30 for sigmoid
	print_period = 10 # print every 10 steps

	# Full 784 dimensionality data set
	X, Y = get_normalized_data()
	lr = 0.00004 # these are pre-computed.. will learn later how to optimize the hyperparameters
	reg = 0.01

	Xtrain = X[:-1000, ]
	Ytrain = Y[:-1000]
	Xtest = X[-1000:, ]
	Ytest = Y[-1000:]
	Ytrain_ind = y2indicator(Ytrain)
	Ytest_ind = y2indicator(Ytest)

	N, D = Xtrain.shape
	batch_sz = 500
	n_batches = int(N / batch_sz)

	M = 300 # number of hidden units, note earlier we had seen that when we do a PCA 300 PCs explain a great variablity of the MNIST data. So, with that in mind we choose 300 hidden units.
	K = 10 # 10 classes
	W1 = np.random.randn(D, M) / np.sqrt(D + K)
	b1 = np.random.randn(M)
	W2 = np.random.randn(M, K) / np.sqrt(M)
	b2 = np.random.randn(K)

	#################### 1. Batch

	LL_batch = []
	CR_batch = []
	for i in range(max_iter):
		for j in range(n_batches):
			Xbatch = Xtrain[j * batch_sz:(j * batch_sz + batch_sz), ]
			Ybatch = Ytrain_ind[j * batch_sz:(j * batch_sz + batch_sz), ]
			pYbatch, Z = forward(Xbatch, W1, b1, W2, b2)

			# updating weights and biases using Gradient Descent-
			W2 -= lr*(derivative_w2(Z, Ybatch, pYbatch) + reg*W2)
			b2 -= lr*(derivative_b2(Ybatch, pYbatch) + reg*b2)
			W1 -= lr*(derivative_w1(Xbatch, Z, Ybatch, pYbatch, W2) + reg*W1)
			b1 -= lr*(derivative_b1(Z, Ybatch, pYbatch, W2) + reg*b1)

			if j % print_period == 0:
				# Calculate just for LL (Log - likelihood)
				pY, _ = forward(Xtest, W1, b1, W2, b2)
				ll = cost(pY, Ytest_ind)
				LL_batch.append(ll)
				print("Batch SGD: Cost at iteration i = %d, j = %d: %f" %(i, j, ll))

				err = error_rate(pY, Ytest)
				CR_batch.append(err)
				print("Error Rate: ", err)

	pY, _ = forward(Xtest, W1, b1, W2, b2)
	print("Final Error Rate: ", error_rate(pY, Ytest)) # Final Error rate is 11 % from the result.

	##################### 2. Batch with REgular Momentum-

	W1 = np.random.randn(D, M) / np.sqrt(D + K)
	b1 = np.random.randn(M)
	W2 = np.random.randn(M, K) / np.sqrt(M)
	b2 = np.random.randn(K)

	LL_momentum = []
	CR_momentum = []

	# Momentum and previous weight changes-
	mu = 0.9
	dW2 = 0
	db2 = 0
	dW1 = 0
	db1 = 0

	for i in range(max_iter):
		for j in range(n_batches):
			Xbatch = Xtrain[j * batch_sz:(j*batch_sz + batch_sz), ]
			Ybatch = Ytrain_ind[j * batch_sz:(j*batch_sz + batch_sz), ]
			pYbatch, Z = forward(Xbatch, W1, b1, W2, b2)

			# Updates-
			dW2 = mu*dW2 - lr*(derivative_w2(Z, Ybatch, pYbatch) + reg*W2)
			W2 += dW2

			db2 = mu*db2 - lr*(derivative_b2(Ybatch, pYbatch) + reg*b2)
			b2 += db2

			dW1 = mu*dW1 - lr*(derivative_w1(Xbatch, Z, Ybatch, pYbatch, W2) + reg*W1)
			W1 += dW1

			db1 = mu*db1 - lr*(derivative_b1(Z, Ybatch, pYbatch, W2) + reg*b1)
			b1 += db1

			if j % print_period == 0:
				# Calculate just for LL
				pY, _ = forward(Xtest, W1, b1, W2, b2)
				ll = cost(pY, Ytest_ind)
				LL_momentum.append(ll)
				print("Regular Momentum Batch SGD: Cost at iteration i = %d, j = %d : %f" % (i, j, ll))

				err = error_rate(pY, Ytest)
				CR_momentum.append(err)
				print("Error Rate: ", err)

	pY, _ = forward(Xtest, W1, b1, W2, b2)
	print("Final Error Rate: ", error_rate(pY, Ytest))  # the Final error rate is 5 %

	############### 3. Batch with Nesterov Momentum-

	W1 = np.random.randn(D, M) / np.sqrt(D + K)
	b1 = np.random.randn(M)
	W2 = np.random.randn(M, K) / np.sqrt(M)
	b2 = np.random.randn(K)

	LL_nest = []
	CR_nest = []
	mu = 0.9
	dW2 = 0
	db2 = 0
	dW1 = 0
	db1 = 0

	for i in range(max_iter):
		for j in range(n_batches):
			Xbatch = Xtrain[j * batch_sz:(j * batch_sz + batch_sz), ]
			Ybatch = Ytrain_ind[j * batch_sz:(j * batch_sz + batch_sz), ]
			pYbatch, Z = forward(Xbatch, W1, b1, W2, b2)

			#Updates -
			dW2 = mu*mu*dW2 - (1 + mu)*lr*(derivative_w2(Z, Ybatch, pYbatch) + reg*W2)
			W2 += dW2

			db2 = mu*mu*db2 - (1 + mu)*lr*(derivative_b2(Ybatch, pYbatch) + reg*b2)
			b2 += db2

			dW1 = mu*mu*dW1 - (1 + mu)*lr*(derivative_w1(Xbatch, Z, Ybatch, pYbatch, W2) + reg*W1)
			W1 += dW1

			db1 = mu*mu*db1 - (1 + mu)*lr*(derivative_b1(Z, Ybatch, pYbatch, W2) + reg*b1)
			b1 += db1

			if j % print_period == 0:
				#calculate just for LL
				pY, _ = forward(Xtest, W1, b1, W2, b2)
				ll = cost(pY, Ytest_ind)
				LL_nest.append(ll)
				print("Nesterov Momentum Batch SGD: Cost at iteration i = %d, j = %d: %f" % (i, j, ll))

				err = error_rate(pY, Ytest)
				CR_nest.append(err)
				print("Error Rate: ", err)

	pY, _ = forward(Xtest, W1, b1, W2, b2)
	print("Final Error Rate: ", error_rate(pY, Ytest)) # Final error rate is 4.9 %

	plt.plot(LL_batch, label = "batch")
	plt.plot(LL_momentum, label = "regular momentum")
	plt.plot(LL_nest, label = "nesterov momentum")
	plt.legend()
	plt.show()

	# Note with Relu: the error rate is as below-
	# Batch SGD : 5 %
	# Regular Momentum : 3 %
	# Nesterov Momentum : 2.9 %


if __name__ == '__main__':
	main()