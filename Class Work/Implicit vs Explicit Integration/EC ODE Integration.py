# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 02:06:56 2021

@author: Nathan McCarley
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------- Part (a) ---------- #
def uprime(u,v):
    return (998*u)+(1998*v)

def vprime(u,v):
    return (-999*u)+(-1999*v)

def ExactSol(xvec):
    uvec = 0
    vvec = 0
    A = np.array([[998,1998],
              [-999,-1999]])
    lambdas,vecs = np.linalg.eig(A)
    return uvec,vvec

x1 = 0
x2 = 0
xvec = np.array([[x1],
                 [x2]])
(uvec,vvec) = ExactSol(xvec)

# ---------- Part (b) ---------- #
def ExplicitIntegrate(xstart,xstop,h,ystart):
    xs = np.arange(xstart,xstop+h,h)
    I = np.array([[1,0],
                  [0,1]])
    C = np.array([[-998,-1998],
                  [999,1999]])
    n = len(xs)-1
    us = np.zeros(n+1)
    vs = np.zeros(n+1)
    us[0] = ystart[0,0]
    vs[0] = ystart[1,0]
    yn = ystart
    i = 0
    while i < n:
         yn = np.dot((I - (h*C)),yn)
         us[i+1] = yn[0,0]
         vs[i+1] = yn[1,0]
         i += 1
    yvec = us,vs
    return xs,yvec

# ---------- Part (c) ---------- #
def ImplicitIntegrate(xstart,xstop,h,ystart):
    xs = np.arange(xstart,xstop+h,h)
    I = np.array([[1,0],
                  [0,1]])
    C = np.array([[-998,-1998],
                  [999,1999]])
    n = len(xs)-1
    us = np.zeros(n+1)
    vs = np.zeros(n+1)
    us[0] = ystart[0,0]
    vs[0] = ystart[1,0]
    yn = ystart
    i = 0
    while i < n:
         yn = np.dot((np.linalg.inv(I + (h*C))),yn)
         us[i+1] = yn[0,0]
         vs[i+1] = yn[1,0]
         i += 1
    yvec = us,vs
    return xs,yvec
    
# ---------- Part (d) ---------- #
xstart = 0
xstop = 4
h = 0.001
ystart = np.array([[1],
                   [0]])
(xvec,yvec) = ExplicitIntegrate(xstart,xstop,h,ystart)
us = yvec[0]
vs = yvec[1]

# Explicit Method
(xvec,yvec) = ExplicitIntegrate(xstart,xstop,h,ystart)
us = yvec[0]
vs = yvec[1]
plt.plot(xvec,us,'o',label="u")
plt.plot(xvec,vs,'o',label="v")
plt.legend()
plt.title("Explicit Method")
plt.show()
# Implicit Method
(xvec,yvec) = ImplicitIntegrate(xstart,xstop,h,ystart)
us = yvec[0]
vs = yvec[1]
plt.plot(xvec,us,'o',label="u")
plt.plot(xvec,vs,'o',label="v")
plt.legend()
plt.title("Implicit Method")
