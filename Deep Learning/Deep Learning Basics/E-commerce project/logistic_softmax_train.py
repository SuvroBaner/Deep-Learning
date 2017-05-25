# This file trains a logistic softmax on the e-commerce data

import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from process import get_data

# Creating the indicator matrix-

def y2indicator(y, K):
	N = len(y)
	ind = np.zeros((N, K))
	for i in range(N):
		ind[i, y[i]] = 1
	return ind
	
# Get the data

X, Y = get_data()
X, Y = shuffle(X, Y)
Y = Y.astype(np.int32)
D = X.shape[1]
K = len(set(Y))

# Setting up train and test

Xtrain = X[:-100]  # everything upto the 100 sample
Ytrain = Y[:-100]
Ytrain_ind = y2indicator(Ytrain, K)  # now Ytrain_ind is an indicator matrix which is needed for the k class

Xtest = X[-100:]  # everything after 100 sample
Ytest = Y[-100:]
Ytest_ind = y2indicator(Ytest, K)  # now Ytrain_ind is an indicator matrix which is needed for the k class

# Initialize the weight

W = np.random.randn(D, K)
b = np.zeros(K)

def softmax(a): # takes in activation
	expA = np.exp(a)
	return expA / expA.sum(axis = 1, keepdims= True) # sum  along row, i.e. axis = 1
	
def forward(X, W, b):
	return softmax(X.dot(W) + b)
	
def predict(P_Y_given_X):
	return np.argmax(P_Y_given_X, axis = 1) # by row
	
def classification_rate(Y, P):
	return np.mean(Y == P)
	
def cross_entropy(T, pY):
	return -np.mean(T*np.log(pY))
	
train_costs = []
test_costs = []
learning_rate = 0.001

for i in range(10000):
	pYtrain = forward(Xtrain, W, b)
	pYtest = forward(Xtest, W, b)
	
	ctrain = cross_entropy(Ytrain_ind, pYtrain)
	ctest = cross_entropy(Ytest_ind, pYtest)
	
	train_costs.append(ctrain)
	test_costs.append(ctest)
	
	# Now let's perform gradient descent
	
	W -= learning_rate*Xtrain.T.dot(pYtrain - Ytrain_ind)
	b -= learning_rate*(pYtrain - Ytrain_ind).sum(axis = 0)
	
	if i % 1000 == 0:
		print(i, ctrain, ctest)
		
print("Final Train Classification Rate: ", classification_rate(Ytrain, predict(pYtrain)))
print("Final Test Classification Rate: ", classification_rate(Ytest, predict(pYtest)))

legend1, = plt.plot(train_costs, label = 'train cost')
legend2, = plt.plot(test_costs, label = 'test cost')
plt.legend([legend1, legend2])
plt.show()