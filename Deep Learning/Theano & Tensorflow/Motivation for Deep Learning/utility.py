# Get the MNIST dataset from Kaggle : https://www.kaggle.com/c/digit-recognizer/data
# This is a Digit Recognizer problem and use the train.csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def get_normalized_data():
    # images are 28 x 28 = 784 pixels gray scale image, flatten into 1 x 794 vector
    print("Reading in transforming data ...")
    df = pd.read_csv('train.csv')
    data = df.as_matrix().astype(np.float32)
    np.random.shuffle(data)
    
    X = data[:, 1:]
    mu = X.mean(axis = 0)
    std = X.std(axis = 0)
    np.place(std, std == 0, 1)
    X = (X - mu) / std # normalize the data
    
    Y = data[:, 0]
    
    return X, Y

def get_transformed_data():
    print('Reading in and transforming data ... ')
    df = pd.read_csv('train.csv')
    data = df.as_matrix().astype(np.float32)
    np.random.shuffle(data)
    
    X = data[:, 1:]
    mu = X.mean(axis = 0)
    X = X - mu # center the data
    pca = PCA()
    Z = pca.fit_transform(X)
    
    Y = data[:, 0]
    
    plot_cumulative_variance(pca)
    
    return Z, Y, pca, mu

def showImages():
    X, Y = get_normalized_data()
    while True:
        for i in range(10):
            x, y = X[Y == i], Y[Y == i]
            N = len(y)
            j = np.random.choice(N) # randomly picking one image from the above lot
            plt.imshow(x[j].reshape(28, 28), cmap = 'gray')
            plt.title(y[j])
            plt.show()
        prompt = input("Quit ? Enter Y: \n")
        if prompt == 'Y':
            break
    return X, Y

def plot_cumulative_variance(pca):
    P = []
    for p in pca.explained_variance_ratio_:
        if len(P) == 0:
            P.append(p)
        else:
            P.append(p + P[-1])
            
    plt.plot(P)
    plt.show()
    return P

def y2indicator(y):
    N = len(y)
    D = len(set(y))
    ind = np.zeros((N, D))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind
    
def forward(X, W, b):
    a = X.dot(W) + b
    expa = np.exp(a)
    y = expa / expa.sum(axis = 1, keepdims = True) # softmax implementation
    return y
    
def cost(p_y, t):
    tot = t * np.log(p_y)
    return -tot.sum()
    
def predict(p_y):
    return np.argmax(p_y, axis = 1) # by row
    
def error_rate(p_y, t):
    prediction = predict(p_y)
    return np.mean(prediction != t)
    
def gradW(t, y, X):
    return X.T.dot(t - y)
    
def gradb(t, y):
    return (t - y).sum(axis = 0)