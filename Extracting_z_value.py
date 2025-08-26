#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 13:13:52 2025

@author: isabellazane
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import curve_fit
import cmath
import math
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
import os

def average_npy_runs(n,fsize,start=1, end=30):
    """
    Load runs Ckw_abs_{n}_all_{idx}.npy for idx in [start..end], skip
    missing files, and return the voxel-wise average.

    Args:
        n (int):  the run‐type number to load (in place of {n}).
        start (int): first index to try (default 1).
        end   (int): last  index to try (default 20).

    Returns:
        np.ndarray: shape (2133,40000) average over all existing runs.
    """
    directory = ''
    sum_arr = None
    count = 0

    for idx in range(start, end+1):
        path = f"{directory}allmags_{n}_{fsize}_PBC_{idx}_long_very.npy"
        if not os.path.exists(path):
            print(f"Run {idx:02d} not found, skipping.")
            continue

        data = np.load(path, mmap_mode='r')
        if sum_arr is None:
            sum_arr = np.zeros_like(data, dtype=complex)

        sum_arr += data
        count += 1

    if count == 0:
        raise RuntimeError("No .npy files found in the given range!")

    avg = sum_arr / count
    print(f"Averaged over {count} runs.")
    np.save(f"averaged_{fsize}_150_long.npy",avg)
    
    return avg


avg_50 = np.load("averaged_50_200_long.npy", mmap_mode='r')
avg_60 = np.load("averaged_60_200_long.npy", mmap_mode='r')
avg_70 = np.load("averaged_70_200_long.npy", mmap_mode='r')
avg_80 = np.load("averaged_80_200_long.npy", mmap_mode='r')
avg_90 = np.load("averaged_90_200_long.npy", mmap_mode='r')
avg_100 = np.load("averaged_100_200_long.npy", mmap_mode='r')

#load one of the kvalues since they are all the same
k_100 = np.load("kvals_200_100_PBC_30_long_very.npy")


def Kmax(data,i):
    window_size = 20
    kernel = np.ones(window_size) / window_size
    smoothed_data = np.convolve(data, kernel, mode='valid')
    k_smoothed = kvals[window_size - 1:]
    #find max 
    max_value = np.max(smoothed_data)
    
    eighty_max = 0.7 * max_value
    
    closest_index = np.abs(smoothed_data - eighty_max).argmin()
    
    closest_point = smoothed_data[closest_index]
    
    k_value = k_smoothed[closest_index]
    
    t = 0.01*i
    
    return k_value, t, closest_index


auto = avg_50
kvals = k_100

numsteps = 40000
kvals0 = np.zeros(numsteps)
tvals = np.zeros(numsteps)
cindex = np.zeros(numsteps)

for i in range(numsteps):
    a,b,c = Kmax(auto[:,i],i)
    kvals0[i] = a
    tvals[i] = b
    cindex[i] = c
    
#cut off the beginning and after T/4 

q =10000
p = 70
kvals2 = kvals0[p:q]
tvals2 = tvals[p:q]

log_k0 = np.log(kvals2)
log_k = savgol_filter(log_k0, window_length=23, polyorder=2)
log_t = np.log(tvals2)

coefficients = np.polyfit(log_t,log_k,1)
slope,intercept = coefficients

print('slope is: ', slope)
print('intercept is: ', intercept)
print('max fluctuation size is: ')
y_fit = slope*log_t + intercept

plt.scatter(log_t,log_k,s=2)
plt.xlabel(r'$\ln(t)$')
plt.ylabel(r'$\ln(k^*)$')
plt.plot(log_t,y_fit,color='red')
#plt.plot(log_t,-0.33*log_t +intercept)
plt.plot()
plt.show()


    