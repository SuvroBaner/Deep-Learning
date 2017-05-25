# This file reads the data from an ecommerce_data flat file which is transactions related
# The csv can be found here : https://github.com/SuvroBaner/machine_learning_examples/tree/master/ann_logistic_extra

import numpy as np
import pandas as pd

def get_data():
	df = pd.read_csv('ecommerce_data.csv')
	data = df.as_matrix()  # turning df into a numpy matrix
	
	X = data[:, :-1] # everything upto the last column
	Y = data[:, -1] # the last column
	
	# We will normalize the numerical column
	X[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std() # for n products viewed
	X[:, 2] = (X[:, 2] - X[:, 2].mean()) / X[:, 2].std() # for visit duration
	
	# Work on the categorical column
	N, D = X.shape
	X2 = np.zeros((N, D+3)) # for time_of_day has 4 different categorical values based on six hours time period
	X2[:, 0:(D-1)] = X[:, 0:(D-1)] # all columns until time of the day
	
	# We will do a one-hot encoding for the other four columns i.e. Dth, D+1th, D+2th, D+3th
	for n in range(N):  # loop through every sample
		t = int(X[n, D-1]) # the value would be either 0, 1, 2, and 3
		X2[n, t+D-1] = 1 # creating the one-hot encoding
		
	# Another way to create this one - hot  encoding is tobytes
	Z = np.zeros((N, 4))  # 4 for four columns
	Z[np.arange(N), X[:, D-1].astype(np.int32)] = 1 # X[:, D-1].astype(np.int32) gives you the value of the levels (500,)
	# X2[:, -4:] = Z
	assert(np.abs(X2[:, -4:] -Z).sum() < 10e-10)
	
	return X2, Y