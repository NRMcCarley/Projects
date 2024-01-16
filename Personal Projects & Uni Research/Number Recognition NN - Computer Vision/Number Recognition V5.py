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

def init_params(n1, n2):
    # Initializes Weights and Biases
    W1 = np.random.rand(n1, 784) - 0.5
    b1 = np.random.rand(n1, 1) - 0.5
    W2 = np.random.rand(n2, n1) - 0.5
    b2 = np.random.rand(n2, 1) - 0.5
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
    dZ2 = 2 * (A2 - Y)
    dW2 = (1 / 10) * np.multiply(dZ2, np.dot(sigmoid_prime(Z2), A1.T))
    db2 = (1 / 10) * np.multiply(dZ2,sigmoid_prime(Z2))
    dZ1 = sigmoid_prime(Z1)
    dW1 = (1 / 10) * np.dot(W2.T, np.dot(dZ1, A0.T))
    db1 = (1 / 10) * np.dot(W2.T, dZ1)
    return dW1, db1, dW2, db2

def update_params(alpha, W1, b1, W2, b2, dW1, db1, dW2, db2):
    W1 = W1 - (alpha * dW1)
    b1 = b1 - (alpha * b1)
    W2 = W2 - (alpha * dW2)
    b2 = b2 - (alpha * b2)
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
hiddenNeurons = 10
outputNeurons = 10
W1, b1, W2, b2 = init_params(hiddenNeurons, outputNeurons)

# Gradient Descent Iterations
k = 0
gradIters = 1000
xs = np.arange(0,gradIters)
Error = np.zeros(gradIters)
j = 0
while k < gradIters:
    #  Find Error for each batch using forward propagation and average it
    dW1Sums = np.zeros((10,784))
    db1Sums = np.zeros((10,1))
    dW2Sums = np.zeros((10,10))
    db2Sums = np.zeros((10,1))
    errorSums = 0
    i = 0 
    numBatches = 500
    if j == 60000:
        j = 0
    else: 
        while i < numBatches:
            temp = trainImgSet[:,j]
            A0 = temp.reshape((784,1))
            trueY = trainLabelSet[0,j]
            Y = np.zeros((10,1))
            Y[trueY] = 1
            Z1, A1, Z2, A2 = forward_prop(A0, W1, b1, W2, b2)
            dW1, db1, dW2, db2 = backward_prop(A0, W1, b1, Z1, A1, W2, b2, Z2, A2, Y)
            dW1Sums = dW1Sums + dW1
            db1Sums = db1Sums + db1
            dW2Sums = dW2Sums + dW2 
            db2Sums = db2Sums + db2
            diff = A2 - Y
            errorSums += np.sum(np.square(diff))
            j += 1
            i += 1
        print("Error for Batch", k+1, ":", errorSums / numBatches)
        Error[k] = errorSums / numBatches
        alpha = 0.01
        W1, b1, W2, b2 = update_params(alpha, W1, b1, W2, b2, dW1, db1, dW2, db2)
        k += 1
    
plt.plot(xs,Error)

j = 0
while j < 100:
    print("Correct Value: ", testLabelSet[0,j])
    temp = testImgSet[:,j]
    A0 = temp.reshape((784,1))
    Z1, A1, Z2, A2 = forward_prop(A0, W1, b1, W2, b2)
    print(np.argmax(A2))
    j += 10