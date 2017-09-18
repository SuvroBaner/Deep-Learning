# Compare RMSProp vs Constant Learning Rate-

import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

from utility import get_normalized_data, error_rate, cost, y2indicator
from mlp import forward, derivative_w2, derivative_b2, derivative_w1, derivative_b1

def main():
	max_iter = 20 # make it 30 for sigmoid
	print_period = 10

	X, Y = get_normalized_data()
	lr = 0.00004
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

	M = 300 # 300 hidden units... we learnt this from the PCA 
	K = 10 # 10 output classes
	W1 = np.random.randn(D, M) / np.sqrt(D + K)
	b1 = np.random.randn(M)
	W2 = np.random.randn(M, K) / np.sqrt(M)
	b2 = np.random.randn(K)

	################ 1. Constant Learning Rate-
	LL_batch = []
	CR_batch = []

	for i in range(max_iter):
		for j in range(n_batches):
			Xbatch = Xtrain[j * batch_sz:(j * batch_sz + batch_sz), ]
			Ybatch = Ytrain_ind[j * batch_sz:(j * batch_sz + batch_sz), ]
			pYbatch, Z = forward(Xbatch, W1, b1, W2, b2)

			# Updates-
			W2 -= lr*(derivative_w2(Z, Ybatch, pYbatch) + reg*W2)
			b2 -= lr*(derivative_b2(Ybatch, pYbatch) + reg*b2)
			W1 -= lr*(derivative_w1(Xbatch, Z, Ybatch, pYbatch, W2) + reg*W1)
			b1 -= lr*(derivative_b1(Z, Ybatch, pYbatch, W2) + reg*b1)

			if j % print_period == 0:
				pY, _ = forward(Xtest, W1, b1, W2, b2)
				ll = cost(pY, Ytest_ind)
				LL_batch.append(ll)
				print("Constant Learning: Cost at iteration i = %d, j = %d : %f" % (i, j, ll))

				err = error_rate(pY, Ytest)
				CR_batch.append(err)
				print("Error rate: ", err)

	pY, _ = forward(Xtest, W1, b1, W2, b2)
	print("Constant Learning: Final Error Rate: ", error_rate(pY, Ytest)) # The final error rate is 6.3 %

	############### 2. RMSprop -
	W1 = np.random.randn(D, M) / np.sqrt(D + K)
	b1 = np.random.randn(M)
	W2 = np.random.randn(M, K) / np.sqrt(M)
	b2 = np.random.randn(K)

	LL_rms = []
	CR_rms = []

	lr0 = 0.001 # initial learning rate. If you set it too high you will get NaN
	
	cache_W2 = 0
	cache_b2 = 0
	cache_W1 = 0
	cache_b1 = 0

	decay_rate = 0.999
	eps = 0.000001

	for i in range(max_iter):
		for j in range(n_batches):
			Xbatch = Xtrain[j * batch_sz:(j * batch_sz + batch_sz), ]
			Ybatch = Ytrain_ind[j * batch_sz:(j * batch_sz + batch_sz), ]
			pYbatch, Z = forward(Xbatch, W1, b1, W2, b2)

			### Updates-
			gW2 = derivative_w2(Z, Ybatch, pYbatch) + reg*W2
			cache_W2 = decay_rate * cache_W2 + (1 - decay_rate) * gW2 * gW2
			W2 -= lr0 * gW2 / (np.sqrt(cache_W2) + eps)

			gb2 = derivative_b2(Ybatch, pYbatch) + reg*b2
			cache_b2 = decay_rate * cache_b2 + (1 - decay_rate) * gb2 * gb2
			b2 -= lr0 * gb2 / (np.sqrt(cache_b2) + eps)

			gW1 = derivative_w1(Xbatch, Z, Ybatch, pYbatch, W2) + reg*W1
			cache_W1 = decay_rate * cache_W1 + (1 - decay_rate) * gW1 * gW1
			W1 -= lr0 * gW1 / (np.sqrt(cache_W1) + eps)

			gb1 = derivative_b1(Z, Ybatch, pYbatch, W2) + reg*b1
			cache_b1 = decay_rate * cache_b1 + (1 - decay_rate) * gb1 * gb1
			b1 -= lr0 * gb1 / (np.sqrt(cache_b1) + eps)

			if j % print_period == 0:
				pY, _ = forward(Xtest, W1, b1, W2, b2)
				ll = cost(pY, Ytest_ind)
				LL_rms.append(ll)
				print("RMSProp : Cost at iteration i = %d, j = %d: %f" % (i, j, ll))

				err = error_rate(pY, Ytest)
				CR_rms.append(err)
				print("Error Rate: ", err)

	pY, _ = forward(Xtest, W1, b1, W2, b2)
	print("Final Error Rate: ", error_rate(pY, Ytest))

	plt.plot(LL_batch, label = "Constant")
	plt.plot(LL_rms, label = "rms")
	plt.legend()
	plt.show()


if __name__ == '__main__':
	main()

