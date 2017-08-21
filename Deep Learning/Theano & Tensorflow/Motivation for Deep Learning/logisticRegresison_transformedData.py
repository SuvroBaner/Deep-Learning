# Get the MNIST dataset from Kaggle : https://www.kaggle.com/c/digit-recognizer/data
# This is a Digit Recognizer problem and use the train.csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from utility import get_transformed_data, showImages, plot_cumulative_variance, y2indicator, forward, cost, predict, error_rate, gradb, gradW

def benchmark_pca():
    X, Y, _, _ = get_transformed_data()  # note here X is transformed to Principal Components
    X = X[:, :300] # only taking 300 PCA s as it almost explains close to 100 % of variance.
    
    # Normalize X first-
    mu = X.mean(axis = 0)
    std = X.std(axis = 0)
    X = (X - mu) / std
    
    print("Performing Logistic Regression ...")
    Xtrain = X[:-1000, :]
    Ytrain = Y[:-1000]
    Xtest = X[-1000:, ]
    Ytest = Y[-1000:]
    
    N, D = Xtrain.shape
    Ytrain_ind = np.zeros((N, 10))
    for i in range(N):
        Ytrain_ind[i, Ytrain[i]] = 1
        
    Ntest = len(Ytest)
    Ytest_ind = np.zeros((Ntest, 10))
    for i in range(Ntest):
        Ytest_ind[i, Ytest[i]] = 1
    
    K = len(set(Ytest))
    W = np.random.randn(D, K) / np.sqrt(D + K)
    b = np.zeros(K)
    
    LL = []
    LLtest = []
    CRtest = []
    
    lr = 0.0001
    reg = 0.01
    
    for i in range(200):
        p_y = forward(Xtrain, W, b)
        ll = cost(p_y, Ytrain_ind)
        LL.append(ll)
        
        p_y_test = forward(Xtest, W, b)
        lltest = cost(p_y_test, Ytest_ind)
        LLtest.append(lltest)
        
        err = error_rate(p_y_test, Ytest)
        CRtest.append(err)
        
        W += lr*(gradW(Ytrain_ind, p_y, Xtrain) - reg*W)
        b += lr*(gradb(Ytrain_ind, p_y) - reg*b)
        if i % 10 == 0:
            print("Cost at iteration %d: %.6f" % (i, ll))
            print("Error rate:", err)

    p_y = forward(Xtest, W, b)
    print("Final error rate:", error_rate(p_y, Ytest))
    iters = range(len(LL))
    plt.plot(iters, LL, iters, LLtest)
    plt.show()
    plt.plot(CRtest)
    plt.show()

if __name__ == '__main__':
    benchmark_pca()