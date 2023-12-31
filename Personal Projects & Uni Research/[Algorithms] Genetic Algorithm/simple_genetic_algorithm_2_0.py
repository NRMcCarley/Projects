# -*- coding: utf-8 -*-
"""Simple Genetic Algorithm 2.0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1EpnI7r_ZpV_Nsgf0dk-M6zjiNT8IbFhx

# **Import Statements**
"""

import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import pandas as pd

"""# **Functions**"""

def F(chromosome):
  fitness = np.sum(chromosome)
  return fitness

def get_Fs():
  # Obtain Fitness Scores
  i = 0
  while i < numChromosomes:
    chromosome = Gs[gen, i, :]
    sampleScore = F(chromosome)
    Fs[gen, i] = sampleScore
    i += 1

def get_selections():
  # Select members for reproduction
  i = 0
  while i < numChromosomes:
    chromosome = Gs[gen, i, :]
    fitness = Fs[gen, i]
    threshold = np.random.rand() * numGenes
    if fitness > threshold:
      selections.append(list(chromosome))
      i += 1
    else:
      i += 1
  numToCross = int(np.round(len(selections)/2)*2)-2
  Ss = np.zeros((generations, numToCross, numGenes))
  i = 0
  while i < numToCross:
    Ss[gen, i, :] = selections[i]
    i += 1
  return Ss, numToCross

def get_Cs():
  # Create crossover gene pool children
  crossedOffspring = np.zeros((numToCross, numGenes))
  i = 0
  while i < numToCross:
    crossPoint = int(np.round(np.random.rand() * numGenes))
    parent1 = np.reshape(Ss[gen, i, :], (1,numGenes))
    parent2 = np.reshape(Ss[gen, i+1, :], (1,numGenes))
    upperLeft = parent1[0,0:crossPoint]
    upperRight = parent2[0,crossPoint:numGenes]
    lowerLeft = parent2[0,0:crossPoint]
    lowerRight = parent1[0,crossPoint:numGenes]
    child1 = np.concatenate((upperLeft, upperRight))
    child2 = np.concatenate((lowerLeft, lowerRight))
    crossedOffspring[i, :] = child1
    crossedOffspring[i+1, :] = child2
    i += 2
  return crossedOffspring

def get_elites():
  # Select elites to be copied and passed down
  numElites = min(numChromosomes-np.shape(crossedOffspring)[0], eliteMax)
  eliteOffspring = np.zeros((numElites, numGenes))
  eliteParentIndices = np.argpartition(Fs[gen,:], -numElites)[-numElites:]
  i = 0
  while i < numElites:
    parentIndex = eliteParentIndices[i]
    eliteOffspring[i,:] = Gs[gen, parentIndex, :]
    i += 1
  return numElites, eliteOffspring

def get_Ms():
  # Create mutated gene pool children
  numToMutate = np.max(numChromosomes - numToCross - numElites, 0)
  mutatedOffspring = np.zeros((numToMutate, numGenes))
  i = 0
  while i < numToMutate:
    parentIndex = int(np.round(np.random.rand()*(numToCross-1)/numToCross))
    mutationPoint = int(np.round(np.random.rand()*(numGenes-1)/numGenes))
    parent = Ss[gen, parentIndex, :]
    temp = parent[mutationPoint]
    if temp == 1:
      parent[mutationPoint] = 0
      mutatedOffspring[i,:] = parent
      i += 1
    else:
      parent[mutationPoint] = 1
      mutatedOffspring[i,:] = parent
      i += 1

"""# **Creating the Placeholder Matrices**"""

targetData = np.array([[1, 1, 1, 1, 1]])

numGenes = np.shape(targetData)[1]   # Number of genes in each chromosome (sample)
numChromosomes = 50   # Number of chromosomes (samples) in a generation
generations = 20

eliteMax = 15

Gs = np.zeros((generations+1, numChromosomes, numGenes))
Fs = np.zeros((generations+1, numChromosomes))
scores = np.zeros((generations+1))

im = plt.imshow(targetData, cmap='Greys', vmin=0, vmax=1)

"""# **Setting up the First Generation**"""

G_0 = np.round(np.random.rand(numChromosomes, numGenes))
Gs[0, :, :] = G_0
plt.figure(figsize = (10,10))
plt.imshow(Gs[0,:,:], cmap='Greys')

"""# **Main Loop**

Select members for further examination

Create a new offspring pool using a single point crossover

Create the "elite" pool which preserves the best solution by copying the n best solutions from the previous generation

Create the mutation pool using the selected parents
"""

plt.figure(figsize = (10,10))
plt.imshow(Gs[0,:,:], cmap='Greys')
plt.title("Generation: 0")
plt.show()

gen = 0
while gen < generations:
  # Obtain Fitness Scores
  get_Fs()
  print(f"Average Score: {np.sum(Fs[gen,:])/numChromosomes}\n")
  scores[gen] = np.sum(Fs[gen,:])/numChromosomes

  # Select members for reproduction
  selections = []
  Ss, numToCross = get_selections()

  # Create crossover gene pool children
  crossedOffspring = get_Cs()

  # Select elites to be copied and passed down
  numElites, eliteOffspring = get_elites()

  # Create mutated gene pool children
  mutatedOffspring = get_Ms()

  #
  GNext = np.concatenate((eliteOffspring, crossedOffspring, mutatedOffspring))
  Gs[gen+1, :, :] = GNext
  plt.figure(figsize = (10,10))
  plt.imshow(Gs[gen+1,:,:], cmap='Greys', vmin=0, vmax=1)
  plt.title(f"Generation: {gen+1}")
  plt.show()
  gen += 1
# Obtain Fitness Scores
i = 0
while i < numChromosomes:
  chromosome = Gs[gen, i, :]
  sampleScore = F(chromosome)
  Fs[gen, i] = sampleScore
  i += 1
print(f"Average Score: {np.sum(Fs[gen,:])/numChromosomes}\n")
scores[gen] = np.sum(Fs[gen,:])/numChromosomes

plt.plot(scores)
plt.title("Average Score at Each Generation")
plt.ylim(0,numGenes+1)
plt.xlim(0,generations)
plt.show()