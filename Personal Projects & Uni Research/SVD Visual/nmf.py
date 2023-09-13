# -*- coding: utf-8 -*-
"""NMF.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1x_n0M5V2kDiELwPJpHTRFMSD5GHFVwWr
"""

import numpy as np
import matplotlib.pyplot as plt

targetImg = np.array([[0, 0, 0, 0, 1],
                      [0, 0, 0, 1, 1],
                      [1, 0, 1, 1, 1],
                      [1, 1, 1, 1, 0],
                      [0, 1, 1, 0, 0]])
plt.imshow(targetImg, cmap='Greys', vmin=0, vmax=1)

"""# **Pre-Built NumPy SVD Function**"""

U, s, Vt = np.linalg.svd(targetImg)
S = np.diag(s)

reconstructed = U @ S @ Vt
plt.imshow(reconstructed, cmap='Greys', vmin=0, vmax=1)

layers = np.zeros((5, 5, 5))
stages = np.zeros((5, 5, 5))

i = 0
while i < 5:
  u_i = np.reshape(U[:, i], (5, 1))
  s_i = s[i]
  v_i = np.reshape(Vt[i, :], (1, 5))
  prod_i = s_i * (u_i @ v_i)
  layers[i, :, :] = prod_i
  plt.imshow(prod_i, cmap='Greys', vmin=0, vmax=1)
  plt.show()
  i += 1

final = layers[0,:,:] + layers[1,:,:] + layers[2,:,:] + layers[3,:,:] + layers[4,:,:]
plt.imshow(final, cmap='Greys', vmin=0, vmax=1)

stages[0,:,:] = layers[0,:,:]
stages[1,:,:] = layers[0,:,:] + layers[1,:,:]
stages[2,:,:] = layers[0,:,:] + layers[1,:,:] + layers[2,:,:]
stages[3,:,:] = layers[0,:,:] + layers[1,:,:] + layers[2,:,:] + layers[3,:,:]
stages[4,:,:] = layers[0,:,:] + layers[1,:,:] + layers[2,:,:] + layers[3,:,:] + layers[4,:,:]

i = 0
while i < 5:
  plt.imshow(stages[i,:,:], cmap='Greys', vmin=0, vmax=1)
  plt.show()
  i += 1

"""# **My Algorithm**"""

AAt = targetImg @ targetImg.T
AtA = targetImg.T @ targetImg