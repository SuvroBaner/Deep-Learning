# Get the MNIST dataset from Kaggle : https://www.kaggle.com/c/digit-recognizer/data
# This is a Digit Recognizer problem and use the train.csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utility import get_normalized_data, showImages, y2indicator, forward, cost, predict, error_rate, gradb, gradW

def benchmark_full():
    X, Y = showImages()  # show images and get X and Y
    
    print('Performing Logistic Regression ...')
    
    Xtrain = X[:-1000, ]
    Ytrain = Y[:-1000]
    Xtest = X[-1000:, ]
    Ytest = Y[-1000: ]
    
    # convert Ytrain and Ytest to (N x K) matrices on indicator variables
    Ytrain_ind = y2indicator(Ytrain)
    Ytest_ind = y2indicator(Ytest)
    
    N, D = Xtrain.shape
    K = len(set(Ytrain))
    
    W = np.random.randn(D, K) / np.sqrt(D + K)  # weights
    b = np.zeros(K)  # bias
    
    LL = [] # log-likelihood
    LLtest = [] # log-likelihood test
    CRtest = [] # Classification Rate test (Error Rate)
    
    learning_rate = 0.00004
    reg = 0.01
    
    for i in range(500):
        p_y = forward(Xtrain, W, b)
        ll = cost(p_y, Ytrain_ind)
        LL.append(ll)
        
        p_y_test = forward(Xtest, W, b)
        lltest = cost(p_y_test, Ytest_ind)
        LLtest.append(lltest)
        
        err = error_rate(p_y_test, Ytest)
        CRtest.append(err)
        
        # Gradient Ascent
        W += learning_rate*(gradW(Ytrain_ind, p_y, Xtrain) - reg*W)
        b += learning_rate*(gradb(Ytrain_ind, p_y) - reg*b)
        
        if i % 10 == 0:
            print('Cost at iteration %d: %.6f' % (i, ll))  # train cost
            print('Error Rate: ', err) # test error rate
            
    p_y = forward(Xtest, W, b) # using the optimal value of W and b after doing the maximization of likelihood
    print("Final Error rate: ", error_rate(p_y, Ytest))
    
    iters = range(len(LL))
    plt.plot(iters, LL, iters, LLtest)
    plt.show()
    plt.plot(CRtest)
    plt.show()

if __name__ == '__main__':
    benchmark_full()