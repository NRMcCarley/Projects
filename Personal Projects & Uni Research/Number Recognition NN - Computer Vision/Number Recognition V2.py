# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 05:12:05 2022

@author: nathanrm
"""

import numpy as np
import idx2numpy
import matplotlib.pyplot as plt

def ReLU(Z):
    # Applies ReLU function
    return np.maximum(0, Z)

def sigmoid(Z):
    # Applies sigmoid function
    A = 1 / (1 + np.exp(-Z))
    return A

def sigmoid_prime(Z):
    # Applies sigmoid prime function
    return np.multiply(sigmoid(Z), (1 - sigmoid(Z)))

def softmax(Z):
    # Applies softmax function
    tempA = np.exp(Z - np.max(Z))
    A = tempA / tempA.sum()
    return A
    
def ReLU_deriv(Z):
    # Returns derivative of ReLU(Z)
    return Z > 0

def init_params():
    # Initializes Weights and Biases
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def forward_prop(A0, W1, b1, W2, b2):
    # Forward Propagation using a ReLU, then a softmax function
    Z1 = (W1 @ A0) + b1
    A1 = sigmoid(Z1)
    Z2 = (W2 @ A1) + b2
    A2 = sigmoid(Z2)
    return Z1, A1, Z2, A2

def backward_prop(A0, W1, b1, Z1, A1, W2, b2, Z2, A2, Y):
    # Backward Propagation 
    m = 10
    # 1 ----- Use PDs to find dC_dW2 ----- #
    dZ2_dW2 = A1 # 10x1
    dZ2_db2 = 1
    dA2_dZ2 = sigmoid_prime(Z2) # 10x1
    dC_dA2 = 2 * (A2 - Y) # 10x1
    dC_dW2 = (1 / m) * dZ2_db2 * np.dot(dZ2_dW2, np.multiply(dA2_dZ2, dC_dA2).T) # 10x10

    # 2 ----- Use PDs to find dC_db2 ----- #
    dC_db2 = (1 / m) * np.multiply(dA2_dZ2, dC_dA2) # 10x1
    
    # 3 ----- Use PDs to find dC_dA1 = dZ2_dA1 * dA2_dZ2 * dC_dA2 ----- #
    dZ2_dA1 = W2 # 10x10
    dA2_dZ2 = sigmoid_prime(Z1) # dC_dA2 already defined above; 10x1
    dC_dA1 = np.dot(dZ2_dA1, np.multiply(dA2_dZ2, dC_dA2)) # 10x1
    
    # 4 ----- Use dC_dA1 to find dC_dW1 = dZ1_dW1 * dA1_dZ1 * dC_dA1 ----- #
    dZ1_dW1 = A0 # 784x1
    dZ1_db1 = 1
    dA1_dZ1 = sigmoid_prime(Z1) # dC_dA1 already defined above; 10x1
    dC_dW1 = (1 / m) * dZ1_db1 * np.dot(np.multiply(dA1_dZ1, dC_dA1), dZ1_dW1.T) # 10x784
    
    # 5 ----- Use PDs to find dC_db1 ----- #
    dC_db1 = (1 / m) * np.multiply(dA1_dZ1, dC_dA1) # 10x1
    return dC_dW1, dC_db1, dC_dW2, dC_db2

def update_params(alpha, W1, b1, W2, b2, dC_dW1, dC_db1, dC_dW2, dC_db2):
    W1 -= alpha * dC_dW1
    b1 -= alpha * dC_db1
    W2 -= alpha * dC_dW2
    b2 -= alpha * dC_db2
    return W1, b1, W2, b2

# Import and format data sets
trainImgSet = np.array(idx2numpy.convert_from_file('data/train-images.idx3-ubyte')) / 255 #X_train
trainImgSet = np.reshape(trainImgSet, (60000,784)).T
trainLabelSet = np.array(idx2numpy.convert_from_file('data/train-labels.idx1-ubyte')) #Y_train
trainLabelSet = np.reshape(trainLabelSet, (60000,1)).T
testImgSet = np.array(idx2numpy.convert_from_file('data/t10k-images.idx3-ubyte')) / 255 #X_test
testImgSet = np.reshape(testImgSet, (10000,784)).T
testLabelSet = np.array(idx2numpy.convert_from_file('data/t10k-labels.idx1-ubyte')) #Y_test
testLabelSet = np.reshape(testLabelSet, (10000,1)).T

# Initilaize Parameters
W1, b1, W2, b2 = init_params()
numImgs = 60000

# Gradient Descent Iterations
k = 0
gradIters = 200
xs = np.arange(0,gradIters)
Error = np.zeros(gradIters)
while k < gradIters:
    #  Find Error for each image using forward propagation and average it
    i = 0 
    dW1Sums = np.zeros((10,784))
    db1Sums = np.zeros((10,1))
    dW2Sums = np.zeros((10,10))
    db2Sums = np.zeros((10,1))
    errorSums = 0
    while i < numImgs:
        temp = trainImgSet[:,i]
        A0 = temp.reshape((784,1))
        trueY = trainLabelSet[0,i]
        Y = np.zeros((10,1))
        Y[trueY] = 1
        Z1, A1, Z2, A2 = forward_prop(A0, W1, b1, W2, b2)
        dC_dW1, dC_db1, dC_dW2, dC_db2 = backward_prop(A0, W1, b1, Z1, A1, W2, b2, Z2, A2, Y)
        dW1Sums = dW1Sums + dC_dW1
        db1Sums = db1Sums + dC_db1
        dW2Sums = dW2Sums + dC_dW2
        db2Sums = db2Sums + dC_db2
        temp = A2 - Y
        errorSums += np.dot(temp.T, temp)
        i += 1
    dW1Sums = (1 / numImgs) * dW1Sums
    db1Sums = (1 / numImgs) * db1Sums
    dW2Sums = (1 / numImgs) * dW2Sums
    db2Sums = (1 / numImgs) * db2Sums
    print("Error for Iteration ", k+1, ": ", errorSums / numImgs)
    Error[k] = errorSums / numImgs
    alpha = 0.15
    W1, b1, W2, b2 = update_params(alpha, W1, b1, W2, b2, dC_dW1, dC_db1, dC_dW2, dC_db2)
    k += 1
    
plt.plot(xs,Error)

j = 0
while j < 10:
    print("Correct Value: ", testLabelSet[0,j])
    temp = testImgSet[:,j]
    A0 = temp.reshape((784,1))
    Z1, A1, Z2, A2 = forward_prop(A0, W1, b1, W2, b2)
    print(np.argmax(A2))
    j += 1