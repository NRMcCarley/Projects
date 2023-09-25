# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 19:26:36 2022

@author: nathanrm
"""

import numpy as np
import matplotlib.pyplot as plt

numDays = 720

mu = 0.01
gamma = 0.02
Omega = 20
sigma = 0.06
cupPrice = 6
delivFee = 3
stickerPrice = 3
cupCost = 0.04
stickerCost = 0.5
wWage = 15
dWage = 10
epsilon = 2
numWorkers = 3
# walkupMax = 300
delivMax = 250

def WMax(cupPrice):
    return 500*np.exp(-1 + -0.5*cupPrice) + 140

def w(t):
    walkupMax = WMax(cupPrice)
    return walkupMax/(1+np.exp(2-(mu*t))) + 2*np.sin(0.5*t) + 40*np.sin(0.01*t)

def d(t):
    return delivMax/(1+np.exp(2.9-(gamma*t)))

def revenue(t):
    return w(t)*cupPrice + d(t)*(cupPrice+delivFee) + sigma*stickerPrice*w(t)
    
def COGS(t):
    return (w(t) + d(t) * cupCost) + sigma*stickerCost*w(t)

def L(t):
    numDrivers = d(t)/(Omega * epsilon)
    return Omega * ((numWorkers * wWage) + (numDrivers * dWage))

def expens(t): 
    return COGS(t) + L(t)
    
def p(t):
    return revenue(t) - expens(t)

ts = np.arange(0,numDays)
walkups = np.zeros(numDays)
deliveries = np.zeros(numDays)
revenues = np.zeros(numDays)
expenses = np.zeros(numDays)
profits = np.zeros(numDays)
totalProfit = np.zeros(numDays)

i = 0
while i < numDays:
    walkups[i] = w(i)
    deliveries[i] = d(i)
    revenues[i] = revenue(i)
    expenses[i] = expens(i)
    profits[i] = p(i)
    totalProfit[i] = totalProfit[i-1] + profits[i]
    if i % 30 == 0:
        profits[i] -= 2500
        totalProfit[i] -= 2500
        i += 1
    else:
        i += 1
    
    
    
DTF = 720
        
plt.plot(ts,totalProfit)
plt.grid()
plt.xlabel('Days')
plt.ylabel('Total Profit')
plt.xlim(0,DTF)
# plt.ylim(-1000000,1000000)
plt.show()

plt.plot(ts,profits)
plt.grid()
plt.xlabel('Days')
plt.ylabel('Daily Profit')
plt.xlim(0,DTF)
plt.ylim(-3000,3000)

print(f"Customers on Day 1: {walkups[1]}")
print(f"Deliveries on Day 1: {deliveries[1]}")

print(f"\nCustomers on Day 7: {walkups[7]}")
print(f"Deliveries on Day 7: {deliveries[7]}")

print(f"\nCustomers on Day 30: {walkups[30]}")
print(f"Deliveries on Day 30: {deliveries[30]}")

print(f"\nCustomers on Day 90: {walkups[90]}")
print(f"Deliveries on Day 90: {deliveries[90]}")

print(f"\nCustomers on Day 180: {walkups[180]}")
print(f"Deliveries on Day 180: {deliveries[180]}")
    