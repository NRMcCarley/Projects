# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 01:11:34 2022

@author: nathanrm
"""

import numpy as np
import idx2numpy
import matplotlib.pyplot as plt

def sigmoid(Z):
    # Applies sigmoid function
    A = 1 / (1 + np.exp(-Z))
    return A

def sigmoid_prime(Z):
    # Applies sigmoid prime function
    return np.multiply(sigmoid(Z), (1 - sigmoid(Z)))

def init_params(n1, n2):
    # Initializes Weights and Biases
    W1 = np.random.rand(n1, 784) - 0.5
    b1 = np.random.rand(n1, 1) - 0.5
    W2 = np.random.rand(n2, n1) - 0.5
    b2 = np.random.rand(n2, 1) - 0.5
    return W1, b1, W2, b2

def gen_Y(YData):
    n = np.shape(YData)[1]
    Y = np.zeros((10, n))
    i = 0
    while i < n:
        trueY = YData[0, i]
        Y[trueY, i] = 1
        i += 1
    return Y

def forward_prop(A0, W1, b1, W2, b2):
    # Forward Propagation
    Z1 = np.dot(W1, A0) + b1   # n1, samples
    A1 = sigmoid(Z1)   # n1, samples
    Z2 = np.dot(W2, A1) + b2   # n2, samples
    A2 = sigmoid(Z2)   # n2, samples
    return Z1, A1, Z2, A2

def back_prop(A0, W1, b1, Z1, A1, W2, b2, Z2, A2, Y):
    # Backwards Propagation
    samples = np.shape(A0)[1]
    dZ2 = (2 * (A2 - Y))
    dW2 = (1 / samples) * np.dot(dZ2, A1.T)
    db2 = (1 / samples) * np.sum(dZ2, 1)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), sigmoid_prime(Z1))
    dW1 = (1 / samples) * np.dot(dZ1, A0.T)
    db1 = (1 / samples) * np.sum(dZ1, 1)
    return dW1, db1, dW2, db2

def update_params(alpha, W1, b1, W2, b2, dW1, db1, dW2, db2):
    W1 = W1 - (alpha * dW1)
    b1 = b1 - (alpha * b1)
    W2 = W2 - (alpha * dW2)
    b2 = b2 - (alpha * b2)
    return W1, b1, W2, b2

def err(A2, Y):
    diff = A2 - Y
    m, n = np.shape(diff)
    error = (1 / (m*n)) * np.sum(np.square(diff))
    return error

def optimize_params(X, Y, alpha, iters, n1, n2):
    A0 = X
    W1, b1, W2, b2 = init_params(n1, n2)
    i = 0
    while i < iters:
        Z1, A1, Z2, A2 = forward_prop(A0, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = back_prop(A0, W1, b1, Z1, A1, W2, b2, Z2, A2, Y)
        W1, b1, W2, b2 = update_params(alpha, W1, b1, W2, b2, dW1, db1, dW2, db2)
        if i % 10:
            i += 1
        else:
            # print(f"Error at Iteration {i+10}: \n{err(A2, Y)}")
            i += 1
    return W1, b1, W2, b2

def get_predictions(X_Test, W1, b1, W2, b2):
    i = 0
    numTests = np.shape(X_Test)[1]
    Expected_Ys = np.zeros((1, numTests))
    while i < numTests:
        X = X_Test[:,i]
        X = np.reshape(X, (784, 1))
        Y = forward_prop(X, W1, b1, W2, b2)[3]
        Expected_Ys[0, i] = np.argmax(Y)
        i += 1
    return Expected_Ys

def get_accuracy(xs, ys):
    numTests = np.shape(xs)[1]
    sums = 0
    i = 0
    while i < numTests:
        x = xs[0, i]
        y = ys[0, i]
        if x == y:
            sums += 1
            i += 1
        else:
            i += 1
    return sums / numTests


# Import and format data sets
trainImgSet = np.array(idx2numpy.convert_from_file("C:/Users/nathanrm/Documents/Coding Projects/Number Recognition NN/data/train-images.idx3-ubyte")) / 255 #X_train
trainImgSet = np.reshape(trainImgSet, (60000,784)).T
trainLabelSet = np.array(idx2numpy.convert_from_file("C:/Users/nathanrm/Documents/Coding Projects/Number Recognition NN/data/train-labels.idx1-ubyte")) #Y_train
trainLabelSet = np.reshape(trainLabelSet, (60000,1)).T
testImgSet = np.array(idx2numpy.convert_from_file("C:/Users/nathanrm/Documents/Coding Projects/Number Recognition NN/data/t10k-images.idx3-ubyte")) / 255 #X_test
testImgSet = np.reshape(testImgSet, (10000,784)).T
testLabelSet = np.array(idx2numpy.convert_from_file("C:/Users/nathanrm/Documents/Coding Projects/Number Recognition NN/data/t10k-labels.idx1-ubyte")) #Y_test
testLabelSet = np.reshape(testLabelSet, (10000,1)).T

# Initialize
X = trainImgSet
Y = gen_Y(trainLabelSet)

# Optimize
n1 = 50
n2 = 10
alpha = 1.5
iters = 500
# W1, b1, W2, b2 = optimize_params(X, Y, alpha, iters, n1, n2)

"""# Test
X_Test = testImgSet
Y_Test = gen_Y(testLabelSet)
predictions = get_predictions(X_Test, W1, b1, W2, b2)
accuracy = get_accuracy(predictions, testLabelSet)
print(f"Accuracy on Test Data with n1 = {n1} and alpha = {alpha}: {accuracy*100}%")"""

errorMat = np.zeros((7,20))
i = 0
N = 80
while N < 100:
    j = 0
    A = 0.1
    while A < 2.1:
        W1, b1, W2, b2 = optimize_params(X, Y, A, 500, N, 10)
        predictions = get_predictions(testImgSet, W1, b1, W2, b2)
        accuracy = get_accuracy(predictions, testLabelSet)
        print(f"Accuracy on Test Data with n1 = {N} and alpha = {A}: {accuracy*100}%")
        errorMat[i,j] = accuracy
        j += 1
        A += 0.1
    i += 1
    N += 10

