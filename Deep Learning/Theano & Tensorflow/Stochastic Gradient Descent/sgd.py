# In this file we compare the progression of the cost function vs iteration for 3 cases-
# a) Full Gradient Descent
# b) Batch Gradient Descent
# c) Stochastic Gradient Descent

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime

from utility import get_transformed_data, forward, error_rate, cost, gradW, gradb, y2indicator

def main():
	X, Y, _, _ = get_transformed_data()
	X = X[:, :300] # taking only 1st 300 columns (as this is a part of the PCA)
	
	# Normalize X first-
	mu = X.mean(axis = 0)
	std = X.std(axis = 0)
	X = (X - mu) / std
	
	print('Performing Logistic Regression')
	Xtrain = X[:-1000, ]
	Ytrain = Y[:-1000]
	Xtest = X[-1000:, ]
	Ytest = Y[-1000:]
	
	N, D = Xtrain.shape
	K = len(set(Ytrain))

	Ytrain_ind = y2indicator(Ytrain)
	Ytest_ind = y2indicator(Ytest)
	
	#### 1. Full Gradient Descent-

	print('--------------------------------------------- Full Gradient Descent --------------------------------------- ')

	W = np.random.randn(D, K) / np.sqrt(D + K)
	b = np.zeros(K)
	LL = []
	lr = 0.0001
	reg = 0.01
	t0 = datetime.now()

	for i in range(200):
		p_y = forward(Xtrain, W, b)

		W += lr*(gradW(Ytrain_ind, p_y, Xtrain) - reg*W)
		b += lr*(gradb(Ytrain_ind, p_y) - reg*b)

		p_y_test = forward(Xtest, W, b)
		ll = cost(p_y_test, Ytest_ind)
		LL.append(ll)

		if i % 10 == 0:
			err = error_rate(p_y_test, Ytest)
			print("Cost at iteration: ", (i, ll))
			print("Error Rate: ", err)

	p_y = forward(Xtest, W, b)
	print("Final error rate: ", error_rate(p_y, Ytest))
	print("Elapsed time for Full Gradient Descent: ", datetime.now() - t0)

	#### 2. Stochastic Gradient Descent - 

	print("-------------------------------------------- Stochastic Gradient Descent -----------------------------------")

	W = np.random.randn(D, K) / np.sqrt(D + K)
	b = np.zeros(K)
	LL_stochastic = []
	lr = 0.0001
	reg = 0.01

	t0 = datetime.now()

	# Only one pass to the data
	for i in range(1): # takes very long since we are computing cost for 41K samples
		tmpX, tmpY = shuffle(Xtrain, Ytrain_ind)
		for n in range(min(N, 500)): # just to make the process short
		#for n in range(N):
			#tmpX, tmpY = shuffle(Xtrain, Ytrain_ind)
			x = tmpX[n, :].reshape(1, D)
			y = tmpY[n, :].reshape(1, K)

			p_y = forward(x, W, b)

			W += lr*(gradW(y, p_y, x) - reg*W)
			b += lr*(gradb(y, p_y) - reg*b)

			p_y_test = forward(Xtest, W, b)
			ll = cost(p_y_test, Ytest_ind)
			LL_stochastic.append(ll)

			#if n % (N/2) == 0:
			err = error_rate(p_y_test, Ytest)
			print("Cost at iteration: ", (n, ll))
			print("Error Rate: ", err)

	p_y = forward(Xtest, W, b)
	print("Final error rate: ", error_rate(p_y, Ytest))
	print("Elapsed time for Stochastic Gradient Descent (SGD): ", datetime.now() - t0)

	#### 3. Batch Gradient Descent -

	print("------------------------------------------------- Batch Gradient Descent ----------------------------------------")

	W = np.random.randn(D, K) / np.sqrt(D + K)
	b = np.zeros(K)
	LL_batch = []
	lr = 0.0001
	reg = 0.01
	batch_sz = 500
	n_batches = int(N / batch_sz)

	t0 = datetime.now()

	for i in range(50):
		tmpX, tmpY = shuffle(Xtrain, Ytrain_ind)
		for j in range(n_batches):
			x = tmpX[j * batch_sz : (j * batch_sz + batch_sz), :]
			y = tmpY[j * batch_sz : (j * batch_sz + batch_sz), :]

			p_y = forward(x, W, b)

			W += lr*(gradW(y, p_y, x) - reg*W)
			b += lr*(gradb(y, p_y) - reg*b)

			p_y_test = forward(Xtest, W, b)
			ll = cost(p_y_test, Ytest_ind)
			LL_batch.append(ll)

			if j % (n_batches/2) == 0:
				err = error_rate(p_y_test, Ytest)
				print("Cost at iteration: ", (i , ll))
				print("Error Rate: ", err)

	p_y = forward(Xtest, W, b)
	print("Final Error Rate: ", error_rate(p_y, Ytest))
	print("Elapsed time for Batch Gradient Descent: ", datetime.now() - t0)

	x1 = np.linspace(0, 1, len(LL))
	plt.plot(x1, LL, label = "full")
	x2 = np.linspace(0, 1, len(LL_stochastic))
	plt.plot(x2, LL_stochastic, label = "stochastic")
	x3 = np.linspace(0, 1, len(LL_batch))
	plt.plot(x3, LL_batch, label = "batch")

if __name__ == '__main__':
    main()
