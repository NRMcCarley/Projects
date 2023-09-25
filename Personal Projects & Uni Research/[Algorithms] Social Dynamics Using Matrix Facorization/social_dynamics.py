# -*- coding: utf-8 -*-
"""Social Dynamics.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10Jy9Xs8HaJDoXJOo6b9YKC5GOYCH2vzd

**0. Imports**
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

"""**1. Data**"""

sheet_url = "https://docs.google.com/spreadsheets/d/15L-7YAya5c7sMaiDBpgRdlWgH4lAHa_pjLci_BAuyKQ/edit#gid=0"
url_1 = sheet_url.replace('/edit#gid=', '/export?format=csv&gid=')

df = pd.read_csv(url_1)
brothers = list(df)[0:40]
df_for_training = df[brothers].astype(float)
R = df_for_training.to_numpy()

"""**2. Functions**"""

def initParams(u, d, k):
  P = np.random.rand(u, k)
  Q = np.random.rand(k, d)
  E = np.zeros((u, d))
  return P, Q, E

def forwardProp(P, Q):
  Rhat = np.dot(P, Q)
  return Rhat

def erFn(Rhat, R):
  tempE = Rhat - R
  errSum = np.sum(tempE**2)
  return errSum, tempE

def backProp(P, Q, E):
  dP = -2 * np.dot(E, Q.T)
  dQ = -2 * np.dot(P.T, E)
  return dP, dQ

def updateParams(P, Q, dP, dQ, alpha):
  P += alpha * dP
  Q += alpha * dQ
  return P, Q

def optimize(P, Q, E, steps):
  h = 0
  while h < steps:
    Rhat = forwardProp(P, Q)
    err, E = erFn(Rhat, R)
    dP, dQ = backProp(P, Q, E)
    P, Q = updateParams(P, Q, dP, dQ, alpha)
    if h % 50 ==0:
      print(f"Error at step {h+1} = {err}")
      h += 1
    else:
      h += 1

def plot_edges2d():
  center = 0
  while center < n:
    test = 0
    point1 = P[center, :]
    while test < n:
      if test == center:
        test += 1
      else:
        point2 = P[test, :]
        x_values = [point1[0], point2[0]]
        y_values = [point1[1], point2[1]]
        if R[test, center] == 2:
          plt.plot(x_values, y_values, linestyle="-", linewidth='0.5', alpha=0.3, color='green')
          test += 1
        elif R[test, center] == 0:
          plt.plot(x_values, y_values, linestyle="-", linewidth='0.5', alpha=0.3, color='red')
          test += 1
        else:
          test += 1
    center += 1

def plot_points2d():
  xdata = P[:, 0]
  ydata = P[:, 1]
  i = 0
  while i < np.shape(P)[0]:
    plt.plot(xdata[i], ydata[i], 'o', label=brothers[i])
    plt.text(xdata[i], ydata[i], brothers[i], color='black', size='6', fontweight='bold')
    i += 1

def twoAxisPlot(P):
  fig = plt.figure()
  ax = plt.axes()
  plot_edges2d()
  plot_points2d()

def plot_edges3d():
  center = 0
  while center < n:
    test = 0
    point1 = P[center, :]
    while test < n:
      if test == center:
        test += 1
      else:
        point2 = P[test, :]
        x_values = [point1[0], point2[0]]
        y_values = [point1[1], point2[1]]
        z_values = [point1[2], point2[2]]
        if R[test, center] == 2:
          plt.plot(x_values, y_values, z_values, linestyle="-", linewidth='0.5', alpha=0.3, color='green')
          test += 1
        elif R[test, center] == 0:
          plt.plot(x_values, y_values, z_values, linestyle="-", linewidth='0.5', alpha=0.3, color='red')
          test += 1
        else:
          test += 1
    center += 1

def threeAxisPlot(P, vert, horz):
  fig = plt.figure()
  ax = plt.axes(projection='3d')
  plot_edges3d()
  xdata = P[:, 0]
  ydata = P[:, 1]
  zdata = P[:, 2]
  i = 0
  while i < np.shape(P)[0]:
    ax.plot3D(xdata[i], ydata[i], zdata[i], 'o', label=brothers[i])
    ax.text(xdata[i], ydata[i], zdata[i], brothers[i], color='black', size='6')
    i += 1
  ax.view_init(vert, horz)

"""**3. Experiments**

2D Graph (2 Features/Groups)
"""

u, d = np.shape(R) # Number of People, u = d
n = u
k = 2 # Features
alpha = 0.001 # Learning Rate
steps = 1000
P, Q, E = initParams(u, d, k)
optimize(P, Q, E, steps)

twoAxisPlot(P)
#plt.xlim(0, 2)
#plt.ylim(0, 2)
plt.show()

"""3D Graph (3 Features/Groups)"""

u, d = np.shape(R) # Number of People
k = 3 # Features
alpha = 0.005 # Learning Rate
steps = 500
P, Q, E = initParams(u, d, k)
optimize(P, Q, E, steps)

vert = 90
horz = 180
while horz > 0:
  threeAxisPlot(P, vert, horz)
  plt.show()
  horz -= 4
  vert -= 2

df2 = pd.DataFrame(data = P.T,
                  columns = brothers)
df3 = pd.DataFrame(data = Q.T,
                  index = brothers)
print("Brother Perception of Friend Groups\n")
print(df2)
print("\n")
print("Group Perceptions of Brothers\n")
print(df3)

"""Markov Analysis"""

def get_stochastic(R):
  n = np.shape(R)[0]
  T = np.zeros((n, n))
  col = 0
  while col < n:
    colSum = np.sum(R[:, col])
    row = 0
    while row < n:
      T[row, col] = (1/colSum) * R[row, col]
      row += 1
    col += 1
  return T

def get_x(T):
  x = (1/n) * np.ones((n, 1))
  return x

def markov(T, x, steps):
  hist = np.zeros((n, steps+1))
  hist[:, 0] = x[0]
  i = 0
  while i < steps:
    temp = np.dot(T, x)
    x = temp
    hist[:, i+1] = x[:, 0]
    i += 1
  return x, hist

def plot_history(hist):
  xs = np.arange(steps+1)
  i = 0
  while i < n:
    plt.plot(xs, hist[i, :], label=brothers[i])
    i += 1
  plt.legend()
  plt.show()

n = np.shape(R)[0]
T = get_stochastic(R)
x = get_x(T)

steps = 10

finalDist, history = markov(T, x, steps)
plot_history(history)
temp = (1/n) * np.sum(finalDist)

n = np.shape(R)[0]
T = np.zeros((n, n))
col = 0
while col < n:
  colSum = np.sum(R[:, col])
  row = 0
  while row < n:
    T[row, col] = (1/colSum) * R[row, col]
    row += 1
  col += 1

print(np.sum(R[:, 0]))
print(np.sum(T))
print(T)
print(x)
print(np.dot(T, x))
print(finalDist)

x = get_x(T)
i = 0
history = np.zeros((n, steps+1))
history[:, 0] = x[0]
while i < steps:
  temp = np.dot(T, x)
  x = temp
  history[:, i+1] = x[:, 0]
  i += 1
print(x)
print(history[:, 1])

temp = np.sum(finalDist)
finalX = finalDist - temp

temp2 = finalDist
i = 0
while i < n:
  maxIndex = np.argmax(temp2)
  print(f"{i+1} - Brother {brothers[maxIndex]}: {finalDist[maxIndex]}")
  temp2[maxIndex] = 0
  i += 1
