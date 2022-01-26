#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 16:03:30 2022

@author: tongchen
Goal of this program is to manually adding random deviation to a given .npy dataset which contains (X;Y;Z) coordinates
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.load("LabeledOriginalMatrix.npy")
#data = np.load("DataForTraining.npy")
dataNew=data

print("datashape",dataNew.shape)

x=dataNew.shape[0]
y=dataNew.shape[1]
z=dataNew.shape[2]

#Gaussian noise is generated for each coordinate element in dataset
#mu is mean & sigma is standard deviation
#Both term can and should be changed to enforce different deviations 
mu, sigma = 0, 0.1 
for i in range(x):
    for j in range(y-1): #y-1 because the last row is to be preserved for 0/1 labeling
        for k in range(z):
            noise = np.random.normal(mu, sigma) 
            dataNew[i,j,k] += noise

#For validation of data deviation 
test = dataNew - data

np.save(
    "DataGaussianNoise",
    dataNew
)