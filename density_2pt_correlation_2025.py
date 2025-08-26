#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 11:24:22 2024

@author: isabellazane
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from mpmath import mp
import pandas as pd
from scipy.fftpack import fft, fftfreq, fftshift
import time
import scipy 
from scipy.fft import fft, fftfreq, fftshift, fft2, fftn
from scipy.fftpack import fft
from scipy.integrate import solve_ivp
import cProfile
from scipy.integrate import odeint
import pstats
from scipy.optimize import curve_fit
import cmath
import math
from numpy.fft import fft, ifft
import itertools
import sys


sqrt_3 = np.sqrt(3)


"""
General code to set up triangular lattice

"""

def T_lattice(num_row,num_col,lattice_spacing = 1,fluctuations = 'yes'):
    #this creates a standard lattice of points
    num_particles = num_row*num_col

    P = []

    if fluctuations == 'no':
        for i in range(num_row):
            for j in range(num_col):
                point = [j*lattice_spacing,i]
                P.append(point)
    if fluctuations == 'yes':
        for i in range(num_row):
            for j in range(num_col):
                random_x = random.uniform(-0.25*lattice_spacing,0.25*lattice_spacing)

                random_y = random.uniform(-0.25*lattice_spacing,0.25*lattice_spacing)
                point = [(j*lattice_spacing)+random_x,i+random_y]

                P.append(point)
    return P




def triangle_maker(n,P):
    #this groups the lattice points into triangles 

    index = []

    triangles = []

    for k in range(n-1):
        for i in range(n-1):
            T1 = [i + (k*n), i + (k*n) + 1, i+n+(k*n)]
            T2 = [i + (k+1)*n, i+1+(k+1)*n, i+1+(k*n)]


            index.append(T1)
            index.append(T2)
    
    #add triangles from periodic boundary conditions
    
    for i in range(n-1):
        Ts1 = [i*n,n*(i+1)-1,n*(i+2)-1]
        Ts2 = [i*n,n*(i+1),n*(i+2)-1]
        Ts3 = [i,(n**2 -n)+i,(n**2 -n)+i+1]
        Ts4 = [i,i+1,(n**2-n)+i+1]
        
        index.append(Ts1)
        index.append(Ts2)
        index.append(Ts3)
        index.append(Ts4)
        
        
    #add the diagonal edge cases
    Te1 = [0,n-1,n**2-n]
    Te2 = [n**2-1,n**2-n,n-1]
    index.append(Te1)
    index.append(Te2)


    if len(P) == n*n:

        for j in range(len(index)):
            Ti = index[j]

            T = [P[Ti[0]],P[Ti[1]],P[Ti[2]]]

            triangles.append(T)

    if len(P) == 2*n*n:
        x = P[:n*n]

        y = P[n*n:]

        new_P = []

        for j in range(n*n):
            new_P.append([x[j],y[j]])


        for j in range(len(index)):
            Ti = index[j]

            T = [new_P[Ti[0]],new_P[Ti[1]],new_P[Ti[2]]]

            triangles.append(T)
    

    return  triangles,index



def Area(T,n):
    T = unwrap_checker(T, n)

    vec1 = [T[0][0] - T[2][0], T[0][1]-T[2][1]]
    vec2 = [T[1][0] - T[2][0], T[1][1]-T[2][1]]

    A = (1/2)*((vec1[0]*vec2[1])-(vec1[1]*vec2[0]))

    return A



def unwrap_coordinate(coord, box_size):
    #Unwrap a coordinate if it spans the boundary.
    return coord % box_size


def unwrap_checker(T,box_size):
    #box size for an NxN lattice is N
    unwrapped_vertices = [(unwrap_coordinate(x, box_size), unwrap_coordinate(y, box_size)) for x, y in T]

    x1, y1 = unwrapped_vertices[0]
    x2, y2 = unwrapped_vertices[1]
    x3, y3 = unwrapped_vertices[2]

    # Check if unwrapping was necessary
    if (x1 - x2) > box_size / 2:
        x2 += box_size
    if (x1 - x3) > box_size / 2:
        x3 += box_size
    if (y1 - y2) > box_size / 2:
        y2 += box_size
    if (y1 - y3) > box_size / 2:
        y3 += box_size
    
    if (x2 - x1) > box_size / 2:
        x1 += box_size
    if (x2 - x3) > box_size / 2:
        x3 += box_size
    if (y2 - y1) > box_size / 2:
        y1 += box_size
    if (y2 - y3) > box_size / 2:
        y3 += box_size
    
    if (x3 - x1) > box_size / 2:
        x1 += box_size
    if (x3 - x2) > box_size / 2:
        x2 += box_size
    if (y3 - y1) > box_size / 2:
        y1 += box_size
    if (y3 - y2) > box_size / 2:
        y2 += box_size
        
    new_T = [[x1,y1],[x2,y2],[x3,y3]]
    
    return new_T

    
    


"""
Single triangle evolution dynamics
"""


def original_area(T_index,n,P0):
    #P0 is the equilibrium lattice

    p1 = P0[T_index[0]]
    p2 = P0[T_index[1]]
    p3 = P0[T_index[2]]

    T = [p1,p2,p3]
    
    newT = unwrap_checker(T, n)

    A0 = Area(newT,n)


    return A0





def ordering_random(num_row,num_col,P):
    triangles = triangle_maker(num_row, P)[0]

    r_list = np.arange(0,len(triangles),dtype = int)
    shuffle = random.shuffle(r_list)

    return r_list


def single_EOM(T,T_index,t,n,P0):
    #evolves a single triangle at a time using the analytic solution
    T = unwrap_checker(T, n)

    A_diff = Area(T,n) - original_area(T_index,n,P0)
  

    c1, c2, c3 = T[0][0], T[1][0], T[2][0]
    b1, b2, b3 = T[0][1], T[1][1], T[2][1]


    x = np.zeros(3)

    y = np.zeros(3)

    sqrt_factor = (1/2)*sqrt_3*A_diff*t

    COS = np.cos(sqrt_factor)
    SIN = np.sin(sqrt_factor)

    const_C = c1+c2+c3
    const_B = b1+b2+b3


    x[0] = (1/3)*((2*c1-c2-c3)*COS-(sqrt_3*(c2-c3)*SIN)+const_C)

    x[1] = (1/3)*(-(c1-2*c2+c3)*COS+(sqrt_3*(c1-c3)*SIN)+const_C)

    x[2] = (1/3)*(-(c1+c2-2*c3)*COS-(sqrt_3*(c1-c2)*SIN)+const_C)

    y[0] = (1/3)*((2*b1-b2-b3)*COS-(sqrt_3*(b2-b3)*SIN)+const_B)

    y[1] = (1/3)*(-(b1-2*b2+b3)*COS+(sqrt_3*(b1-b3)*SIN)+const_B)

    y[2] = (1/3)*(-(b1+b2-2*b3)*COS-(sqrt_3*(b1-b2)*SIN)+const_B)
    
    x = unwrap_coordinate(x, n)
    y = unwrap_coordinate(y, n)


    triangle = [[x[0],y[0]],[x[1],y[1]],[x[2],y[2]]]

    return triangle




def dynamics(h,num_steps,initial_P,num_row,num_col,EQ):

    start_time = time.time()
    #h is how long each individual triangle is run for

    #in a single "step" each triangle is moved. So say there are N

    # triangles. Then num_steps is the number of times every triangle is moved

    #make triangles

    all_T0,index_allT = triangle_maker(num_row, initial_P)

    N = len(all_T0)

    #print("number of triangles: ",N)

    #in a single step every triangle is moved so this is how many times
    #each triangle is being moved. To make things faster we only record the positions
    #of the particles once every triangle is moved


    #number of particles

    n = num_row*num_col

    y = np.zeros((num_steps,n,2))

    #initial Areas

    areas = np.zeros((num_steps,N))

    for i in range(N):
        areas[0][i] = Area(all_T0[i],n)

    #difference in position


    #first step



    y[0] = initial_P

    order = ordering_random(num_row,num_col,initial_P)



    for i in range(1,num_steps):
        #print(box_checker(y[i],num_row))

        y[i] = y[i-1]

        # iterate over all triangles

        for j in range(N):
            k = order[j]
            #find the triangle
            T_index = index_allT[k]


            p1 = y[i][T_index[0]]
            p2 = y[i][T_index[1]]
            p3 = y[i][T_index[2]]

            single_triangle = [p1,p2,p3]


            #evolve the triangle

            new_triangle = single_EOM(single_triangle,T_index,h,num_row,EQ)


            #update the coordinates

            y[i][T_index[0]] = new_triangle[0]
            y[i][T_index[1]] = new_triangle[1]
            y[i][T_index[2]] = new_triangle[2]


        #after evolving all triangles, calculate the new areas of the new triangles

        #first we have to make the new triangles

        new_allT,_ = triangle_maker(num_row, y[i])

        for w in range(N):

            areas[i][w] = Area(new_allT[w],n)






    #now need to separate the X and Y coordinates for plotting purposes

    X = y[:, :, 0].copy()
    Y = y[:, :, 1].copy()


    t = time.time()-start_time
    print("dyanmics,", t)


    return X,Y,y,areas



"""
density 2-point correlation
"""

def run_dynamics(n):
    h = 0.01
    
    num_steps = 100
    
    IC = T_lattice(n, n,fluctuations = 'yes')
    
    EQ = T_lattice(n,n,fluctuations = 'no')
    
    X,Y,positions, areas = dynamics(h, num_steps, IC, n, n, EQ)
    
    
def displacements(P,n):
    EQ = T_lattice(n,n,fluctuations = 'no')
    
    shapes = P.shape
    
    D = np.zeros((shapes[0],shapes[1],shapes[2]))
    
    D_abs = np.zeros((shapes[0],shapes[1]))
    
    for i in range(shapes[0]):
        for j in range(shapes[1]):
            
            dx = P[i][j][0] - EQ[j][0]
            dy = P[i][j][1] - EQ[j][1]

            # Apply periodic boundary conditions
            if abs(dx) > n / 2:
                dx -= np.sign(dx) * n
            if abs(dy) > n / 2:
                dy -= np.sign(dy) * n

            
            

            D[i][j][0] = dx
            D[i][j][1] = dy
            D_abs[i][j] = np.sqrt(dx**2 + dy**2)
    
    return D, D_abs

    
    

def density2pt(kx,ky,positions,num_steps):
    #the positions should be positions for all time 
    
    start_time = time.time()
    
    rho = np.zeros(num_steps,dtype = complex)
    for i in range(num_steps):
        rho[i] = np.sum(np.exp(1j * (kx * positions[i,:, 0] + ky * positions[i,:, 1])))
        
    return rho


def densitysametime(kx,ky,positions,n,t):
    #Get the latest positions
    
    P = positions[t]
    
    
    num_particles = n*n
    
    rho = 0.0j
    
    for i in range(num_particles):
        rho += np.exp(1j * (kx * P[i][0] + ky * P[i][1]))
        
    return rho
    


def density_correlator_plot(n):
    #first get the positions 
    
    h = 0.01
    
    num_steps = 1000
    
    IC = T_lattice(n,n,fluctuations = 'yes')
    
    EQ = T_lattice(n, n, fluctuations = 'no')

    X,Y, positions, areas = dynamics(h, num_steps, IC, n, n, EQ) 
    
    kx = (3*np.pi)/5
    ky = (3*np.pi)/5
    
    D,D_abs = displacements(positions, n)
    
    rho = density2pt(kx,ky,D,num_steps)

    #rho = rho - np.mean(rho)
    
    autocorrelation = np.correlate(rho, rho, mode='full')[num_steps-1:]
   
    tf = int(h*num_steps)
    
    t = np.linspace(0,tf,num_steps)
    
    norm_autocorrelation = autocorrelation/np.abs(autocorrelation[0])
    

    real_autocorrelation = np.real(autocorrelation)/np.abs(autocorrelation[0])
    
    imag_autocorrelation = np.imag(autocorrelation)/np.abs(autocorrelation[0])
    
    #mag = np.sqrt(real_autocorrelation**2 + imag_autocorrelation**2)
    mag = np.abs(norm_autocorrelation)

    plt.plot(t,real_autocorrelation,color = 'blue')
    plt.plot(t,imag_autocorrelation,color ='orange')
    plt.plot(t,mag,color = 'red')
    plt.show()
    
    
    


def density_correlator(n):
    #first get the positions 
    
    start_time = time.time()
    
    h = 0.01
    
    num_steps = 10000
    
    IC = T_lattice(n,n,fluctuations = 'yes')
    
    EQ = T_lattice(n, n, fluctuations = 'no')

    X,Y, positions, areas = dynamics(h, num_steps, IC, n, n, EQ) 
    
    tf = int(h*num_steps)
    
    t = np.linspace(0,tf,num_steps)
    
    kx = (3*np.pi)/5
    ky = (3*np.pi)/5
    
    rho = density2pt(kx,ky,positions,num_steps)
    
    autocorrelation = np.correlate(rho, rho, mode='full')[num_steps-1:]
    
    norm_autocorrelation = autocorrelation/np.abs(autocorrelation[0])

    real_autocorrelation = np.real(autocorrelation)/np.abs(autocorrelation[0])
    
    imag_autocorrelation = np.imag(autocorrelation)/np.abs(autocorrelation[0])
    
    mag = np.abs(norm_autocorrelation)
    
    np.save(f'full_autocorrelate_{n}.npy',norm_autocorrelation)
    np.save(f'real_autocorrelate_{n}.npy',real_autocorrelation)
    np.save(f'imag_autocorrelate_{n}.npy',imag_autocorrelation)
    np.save(f'mag_autocorrelate_{n}.npy',mag)
    
    t = time.time() - start_time
    
    print("autocorrelation: ", t)
    
    
def T_lattice_fluctuations(num_row,num_col,lattice_spacing = 1,fluctuation_size = 0.25):
    num_particles = num_row*num_col
    
    P = []

    for i in range(num_row):
        for j in range(num_col):
            random_x = random.uniform(-fluctuation_size*lattice_spacing,fluctuation_size*lattice_spacing)
                
            random_y = random.uniform(-fluctuation_size*lattice_spacing,fluctuation_size*lattice_spacing)
            point = [(j*lattice_spacing)+random_x,i+random_y]
                
            P.append(point)
            
    return P

    
    
def density_k_vals(kx,ky,positions,h,num_steps):
        tf = int(h*num_steps)
        
        t = np.linspace(0,tf,num_steps)
        
        rho = density2pt(kx,ky,positions,num_steps)
        
        autocorrelation = np.correlate(rho, rho, mode='full')[num_steps-1:]
        
        norm_autocorrelation = autocorrelation/np.abs(autocorrelation[0])
        
        mag = np.abs(norm_autocorrelation)
        
        return mag
    

def positions(n,h,num_steps,fsize):
    IC = T_lattice_fluctuations(n, n,fluctuation_size=fsize)
    
    EQ = T_lattice(n,n,fluctuations='no')
    

    X,Y,y,areas = dynamics(h, num_steps, IC, n, n, EQ)
    
    return y
    
    

def mag(kx,ky):
    return np.sqrt(kx**2 + ky**2)



def pairs(n):
    N = np.arange(1,n+1)
    
    p = list(itertools.product(N, repeat=2))
    
    nump = len(p)
    print(nump)
    mags = np.zeros(nump)
    
    mags = np.array([mag(a, b) for a, b in p])
    
    unique_k = np.unique(mags)
    print(unique_k)
    num_K = len(unique_k)
    
    kvals = np.zeros((nump,num_K,2))
    
    print(kvals.shape)
    
    #first row is equal to the magnitude of k 
    
    for i in range(nump):
        a,b = p[i]
        
        mag_i = mag(a,b)
        
        index_i = np.where(unique_k == mag_i)[0][0]
        
        j = 0
        
        while np.any(kvals[j, index_i] != 0):  
            j += 1
            if j >= nump:
                print("Warning: j exceeded nump, exiting loop.")
                break

        kvals[j, index_i, 0] = a
        kvals[j, index_i, 1] = b
        
   
        
    
    
    return kvals,num_K,unique_k

        


def density_k_total(n,fsize,h=0.01,num_steps=10000):
    P = positions(n,h,num_steps,fsize)
    
    kvals, ksteps,unique_k = pairs(int(n/2))
    
    autos = np.zeros((ksteps,num_steps))
    
    
    for i in range(ksteps):
        K = kvals[:,i,:]
        #find number of non-zero
        s = zero_counter(K)
        
        newK = K[:s,:]
        
        num_small_k = len(newK)
        
        auto_i = []
        
        factor = (2*np.pi)/n
        
        for j in range(num_small_k):
            kx = factor*newK[j][0]
            ky = factor*newK[j][1]
            auto_i.append(density_k_vals(kx,ky,P,h,num_steps))
        
        stack_auto = np.vstack(auto_i)
        autos[i] = np.mean(stack_auto,axis=0)
  
    param = 100*fsize
    np.save(f'allmags_{n}_{param}_PBC_fsize_long.npy',autos)
    np.save(f'kvals_{n}_{param}_PBC_fsize_long.npy',unique_k)
    
    print("done!")


def density_k_total_index(n,index,fsize,h=0.01,num_steps=40000):
    P = positions(n,h,num_steps,fsize)
    
    kvals, ksteps,unique_k = pairs(int(n/2))
    
    autos = np.zeros((ksteps,num_steps))
    
    
    for i in range(ksteps):
        K = kvals[:,i,:]
        #find number of non-zero
        s = zero_counter(K)
        
        newK = K[:s,:]
        
        num_small_k = len(newK)
        
        auto_i = []
        
        factor = (2*np.pi)/n
        
        for j in range(num_small_k):
            print("Running step ",j, "out of ",num_small_k)
            kx = factor*newK[j][0]
            ky = factor*newK[j][1]
            auto_i.append(density_k_vals(kx,ky,P,h,num_steps))
        
        stack_auto = np.vstack(auto_i)
        autos[i] = np.mean(stack_auto,axis=0)
  
    param = int(100*fsize)
    results_directory = "/scratch/alpine/isza5860/run1"
    np.save(results_directory + "/"+f'allmags_{n}_{param}_PBC_{index}_long_very.npy',autos)
    np.save(results_directory + "/"+f'kvals_{n}_{param}_PBC_{index}_long_very.npy',unique_k)
    
    print("done!")
    
    
        
if __name__ == "__main__":
    n = int(sys.argv[1])  # Get 'n' from command-line argument
    index = int(sys.argv[2])
    fsize = float(sys.argv[3])
    density_k_total_index(n,index,fsize)


    