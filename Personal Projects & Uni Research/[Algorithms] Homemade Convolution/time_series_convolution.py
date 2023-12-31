# -*- coding: utf-8 -*-
"""Time Series Convolution.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11qr0rdqyl1SgoD8tiDUf6wpVb7H95OtA
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

def convolve(X, K):

  input_m, input_n = np.shape(X)
  k_m, k_n = np.shape(K)
  output_m = input_m - k_m + 1
  output_n = input_n - k_n + 1
  output_matrix = np.zeros((output_m, output_n))
  i = 0
  while i < output_m:
    # Iterate through the depth of the output matrix
    j = 0
    while j < output_n:
      # Iterate through the width of the output matrix
      a = 0
      k_sums = 0
      while a < k_m:
        # Iterate through the depth of the kernel
        b = 0
        while b < k_n:
          # Iterate through the width of the kernel
          k_sums += (X[a+i, b+j] * K[a, b])
          b += 1
        a += 1
      output_matrix[i, j] = k_sums
      j += 1
    i += 1
  return output_matrix

def relu(X):
  m, n = np.shape(X)
  temp = np.zeros((m, n))
  i = 0
  while i < m:
    j = 0
    while j < n:
      if X[i, j] == 0:
        temp[i, j] = 0
        j += 1
      else:
        temp[i, j] = X[i, j]
        j += 1
    i += 1
  return temp

def sigmoid(X):
  m, n = np.shape(X)
  temp = np.zeros((m, n))
  i = 0
  while i < m:
    j = 0
    while j < n:
      temp[i, j] = 1/(1+np.exp(-X[i, j]))
      j += 1
    i += 1
  return temp

t = 100
T = np.arange(0, t).reshape((1, t))
num_series = 10
X_0 = relu(np.random.rand(num_series, t))

K_1 = np.array([[-2, 4, -2]])
X_1 = relu(convolve(X_0, K_1))

K_2 = np.array([[-2, 0, 1],
                [-2, 0, 1],
                [-2, 0, 1]])
X_2 = relu(convolve(X_1, K_1))

plt.matshow(X_0, cmap=plt.cm.binary)
plt.matshow(X_1, cmap=plt.cm.binary)
plt.matshow(X_2, cmap=plt.cm.binary)

df = pd.read_csv('/content/drive/MyDrive/School/PHYS 295/TWTR.csv')
print(df.head()) #7 columns, including the Date.

tempT = df.to_numpy().T
T = (tempT[1:7,:].astype(float))
m, n = np.shape(T)
j = 0
temp2 = np.zeros((m, n-1))
while j < np.shape(temp2)[1]:
  temp_col = (T[:, j+1] - T[:, j])/T[:, j]
  temp2[:, j] = temp_col
  j += 1

X_0 = sigmoid(temp2*100)
plt.matshow(X_0, cmap=plt.cm.coolwarm)

X_1 = sigmoid(convolve(X_0, K_1))
plt.matshow(X_1, cmap=plt.cm.coolwarm)

X_2 = sigmoid(convolve(X_1, K_1))
plt.matshow(X_2, cmap=plt.cm.coolwarm)

xs = np.arange(np.shape(X_2)[1])
ys = X_2.T
plt.show()
plt.plot(xs, ys)