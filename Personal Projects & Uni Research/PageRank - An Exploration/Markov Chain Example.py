# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 13:14:05 2022

@author: nathanrm
"""

import numpy as np
import matplotlib.pyplot as plt

def generateT(n):
    # Generates an nxn transition matrix with 0s on the diagonal
    i = 0
    randomT = np.zeros((n,n))
    while i < n:
        j = 0
        while j < n:
            if i == j:
                j += 1
            else:
                randomT[i,j] = np.random.randint(101)
                j += 1
        i += 1
    # Sum each column
    j = 0
    colSums = np.zeros(n)
    while j < n:
        i = 0
        while i < n:
            colSums[j] += randomT[i,j]
            i += 1
        j += 1
    # Normalize each column
    i = 0
    while i < n:
        j = 0
        while j < n:
            if i == j:
                j += 1
            else:
                randomT[i,j] /= colSums[j]
                j += 1
        i += 1
    # temp = np.round(randomT,2)
    return randomT


def generateP_0(n):
    # Generate initial nx1 matrix
    P_0 = np.zeros((n,1))
    i = 0
    colSum = 0
    while i < n:
        P_0[i,0] = np.random.randint(0,11)
        colSum += P_0[i,0]
        i += 1
    # Normalize the vector
    i = 0
    while i < n:
        P_0[i,0] /= colSum
        i += 1
    # temp = np.round(P_0,2)
    return P_0


def Grapher(T,P_0,n,alpha,steps):
    # Transcribe first column as P_0 for Step 0
    data = np.zeros((n,steps+1))
    i = 0
    while i < n:
        data[i,0] = P_0[i,0]
        i += 1
    # Transcribe remaining columns with damping factor incorporated
    switchProb = (1/(n-1))*np.ones((n,n))
    np.fill_diagonal(switchProb,0.0)
    transitionMatrix = ((1 - alpha)*T) + (alpha*switchProb)
    print(f"Transition Matrix: \n{transitionMatrix}")
    j = 1
    while j < steps+1:
        i = 0
        tempCol = np.dot(transitionMatrix,data[:,j-1])
        while i < n:
            data[i,j] = tempCol[i]
            i += 1
        if j == steps:
            print(f"Final Distribution: \n{data[:,steps]}")
            j += 1
        else: 
            j += 1
    xs = np.arange(0,steps+1)
    k = 0
    while k < n:    
        plt.plot(xs,data[k,:],label=k+1)
        k += 1
    plt.xlabel('Step')
    plt.ylabel('Fraction')
    # plt.ylim(0,0.5)
    # plt.legend()
    

n = 6
steps = 5
T = generateT(n)
P_0 = generateP_0(n)
alpha = 0.1
Grapher(T,P_0,n,alpha,steps)