# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 15:06:55 2022

@author: nathanrm
"""

import numpy as np
import idx2numpy
import mnist
import matplotlib.pyplot as plt


def ReLU(Z):
    # Applies ReLU function
    return np.maximum(0, Z)

def sigmoid(Z):
    # Applies sigmoid function
    A = 1 / (1 + np.exp(-Z))
    return A

def softmax(Z):
    # Applies softmax function
    tempA = np.exp(Z - np.max(Z))
    A = tempA / tempA.sum()
    return A
    
def ReLU_deriv(Z):
    # Returns derivative of ReLU(Z)
    return Z > 0

def init_params(size):
    # Initializes Weights and Biases
    W1 = np.random.rand(10, size) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def forward_prop(X, W1, b1, W2, b2):
    # Forward Propagation using a ReLU, then a softmax function
    Z1 = (W1 @ X) + b1
    A1 = ReLU(Z1)
    Z2 = (W2 @ A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def backward_prop(X, Y, A1, A2, W2, Z1, m):
    # Backward Propagation 
    dZ2 = 2 * (A2 - Y)
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2Temp = (1 / m) * np.sum(dZ2, 1)
    db2 = np.reshape(db2Temp, (10,1))
    dZ1 = np.dot(W2.T, dZ2) * ReLU_deriv(Z1)
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    dB1Temp = (1 / m) * np.sum(dZ1, 1)
    db1 = np.reshape(dB1Temp, (10,1))
    return dW1, db1, dW2, db2

def update_Wb(alpha, W1, b1, W2, b2, dW1, db1, dW2, db2):
    # Update Weights and Biases
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def make_predictions(X, W1 ,b1, W2, b2):
    Z1, A1, Z2, A2 = forward_prop(X, W1, b1, W2, b2)
    predictions = get_predictions(A2)
    return predictions

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y)/np.size(Y)

def gradient_descent(X, Y, alpha, iterations):
    size, m = np.shape(X)

    W1, b1, W2, b2 = init_params(size)
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(X, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = backward_prop(X, Y, A1, A2, W2, Z1, m)
        W1, b1, W2, b2 = update_Wb(alpha, W1, b1, W2, b2, dW1, db1, dW2, db2)   
        if (i+1) % int(iterations/10) == 0:
            print(f"Iteration: {i+1} / {iterations}")
            prediction = get_predictions(A2)
            print(f'{get_accuracy(prediction, Y):.3%}')
    return W1, b1, W2, b2

def show_prediction(index,X, Y, W1, b1, W2, b2):
    # None => cree un nouvel axe de dimension 1, cela a pour effet de transposer X[:,index] qui un np.array de dimension 1 (ligne) et qui devient un vecteur (colonne)
    #  ce qui correspond bien a ce qui est demande par make_predictions qui attend une matrice dont les colonnes sont les pixels de l'image, la on donne une seule colonne
    vect_X = X
    prediction = make_predictions(vect_X, W1, b1, W2, b2)
    label = Y[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    """current_image = vect_X.reshape((28, 28)) * 255

    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()"""


# Importing Images and Labels for Training sets and Development sets
T_I = np.array(idx2numpy.convert_from_file('data/train-images.idx3-ubyte')) / 255 #X_train
T_I = np.reshape(T_I, (60000,784)).T
T_L = np.array(idx2numpy.convert_from_file('data/train-labels.idx1-ubyte')) #Y_train
T_L = np.reshape(T_L, (60000,1))
# T_L = np.reshape(T_I, (60000,784,1))
D_I = np.array(idx2numpy.convert_from_file('data/t10k-images.idx3-ubyte')) / 255 #X_test
D_I = np.reshape(D_I, (10000,784)).T
D_L = np.array(idx2numpy.convert_from_file('data/t10k-labels.idx1-ubyte')) #Y_test
D_L = np.reshape(D_L, (10000,1))
    
# Setting up the correct values for a test of the image at a certain index
index = 0
trueY = T_L[index]
Y = np.zeros((10,1))
Y[trueY] = 1
X = T_I[index] #X
m = 28
n = 28
alpha = 0.15
iterations = 200
W1, b1, W2, b2 = gradient_descent(T_I, T_L, alpha, iterations)
show_prediction(0,D_I, D_L, W1, b1, W2, b2)
"""
show_prediction(1,D_I, D_L, W1, b1, W2, b2)
show_prediction(2,D_I, D_L, W1, b1, W2, b2)
show_prediction(100,D_I, D_L, W1, b1, W2, b2)
show_prediction(200,D_I, D_L, W1, b1, W2, b2)"""





"""
# Creating random Weights and Biases
W1, b1, W2, b2 = init_params()

# Forward Propagation
Z1, A1, Z2, A2 = forward_prop(X, W1, b1, W2, b2)

# Backwards Propagation
dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)

# Update Weights and Biases with learning rate alpha
W1, b1, W2, b2 = update_Wb(alpha, W1, b1, W2, b2, dW1, db1, dW2, db2)

# code from the correct doc
W1, b1, W2, b2 = gradient_descent(X, Y_train, 0.15, 200)
"""

"""
arr is now a np.ndarray type of object of shape 60000, 28, 28
print(np.shape(T_I))
plt.imshow(T_I[5], cmap=plt.cm.binary)


numTrainIm = 7
a = T_I[1:numTrainIm]
m, n = np.shape(T_I[0])
np.random.shuffle(T_I)
"""