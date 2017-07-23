from __future__ import print_function, division
import pandas as pd
import numpy as np


def getData(balanced_ones = True):
	# images are 48x48 = 2304 size vectors
	# N = 35887
	Y = []
	X = []
	first = True
	for line in open('fer2013.csv'):
		if first:
			first = False
		else:
			row = line.split(',')
			Y.append(int(row[0]))
			X.append([int(p) for p in row[1].split()])

	X, Y = np.array(X) / 255.0, np.array(Y)

	if balanced_ones:
		# balance the 1 class
		X0, Y0 = X[Y!=1, :], Y[Y!=1]
		X1 = X[Y==1, :]
		X1 = np.repeat(X1, 9, axis = 0)
		X = np.vstack([X0, X1])
		Y = np.concatenate((Y0, [1]*len(X1)))

	return X, Y

def getBinaryData():
	Y = []
	X = []
	first = True
	for line in open('fer2013.csv'):
		if first:
			first = False
		else:
			row = line.split(',')
			y = int(row[0])
			if y == 0 or y == 1:
				Y.append(y)
				X.append([int(p) for p in row[1].split()])
	return np.array(X) / 255.0 , np.array(Y)

def sigmoid(A):
	return 1 / (1 + np.exp(-A))

def relu(x):
	return x * (x > 0)

def sigmoid_cost(T, Y):
	return -(T*np.log(Y) + (1-T)*np.log(1-Y)).sum()

def error_rate(targets, predictions):
	return np.mean(targets != predictions)
	
def softmax(A):
	expA = np.exp(A)
	return expA / expA.sum(axis = 1, keepdims = True)
	
def cost(T, Y):
	return -(T*np.log(Y)).sum()
	
def y2indicator(y):
	N = len(y)
	K = len(set(y))
	ind = np.zeros((N, K))
	for i in range(N):
		ind[i, y[i]] = 1
	return ind

def relu(x):
    return x * (x > 0)
	
def cost2(T, Y):
    # same as cost(), just uses the targets to index Y
    # instead of multiplying by a large indicator matrix with mostly 0s
    N = len(T)
    return -np.log(Y[np.arange(N), T]).mean()
