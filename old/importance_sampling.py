#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 11:28:37 2018

@author: owenmadin
"""

import csv
import numpy as np
import random as rm
import pandas as pd
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.stats import distributions
from scipy.optimize import leastsq
from pymc3.stats import hpd
from mpl_toolkits.mplot3d import Axes3D

def f(x):
    f=np.exp(-x**2)
    return f
#x=np.linspace(0,1,num=100)
beta=distributions.beta(1,2)
unif=distributions.uniform(0,1)
#y=beta.pdf(x)
#plt.plot(x,y)
N=10000
I_estimate=[]
I_squared=[]
for i in range(N):
    x=beta.rvs()
    sample=f(x)/beta.pdf(x)
    I_estimate.append(sample)
    I_squared.append(sample**2)
    
    
    
I_estimate=np.asarray(I_estimate)
I_squared=np.asarray(I_squared)


I_hat=np.mean(I_estimate)
I_hat_stdev=np.sqrt((1/N)*(np.mean(I_squared)-I_hat**2))
confidence_interval=(I_hat-I_hat_stdev*1.96,I_hat+I_hat_stdev*1.96)
print(I_hat)
print(I_hat_stdev)
print(confidence_interval)
integral_f=integrate.quad(f,0,1)
print(integral_f)



#%% Antithetic
'''
I_1=[]
I_2=[]
unif=distributions.uniform(0,1)
for i in range(int(N)):
    u=unif.rvs()
    x=-np.log(1-u)
    sample=f(x)/beta.pdf(x)
    I_estimate.append(sample)
    ''' 
I_estimate=[]
I_squared=[]
for i in range(int(N/2)):
    u=unif.rvs()
    x=beta.ppf(u)
    y=beta.ppf(1-u)
    sample_x=f(x)/beta.pdf(x)
    sample_y=f(y)/beta.pdf(y)
    I_estimate.append(sample_x)
    I_estimate.append(sample_y)
    I_squared.append(sample_x**2)
    I_squared.append(sample_y**2)

I_estimate=np.asarray(I_estimate)

I_squared=np.asarray(I_squared)
I_hat=np.mean(I_estimate)
I_hat_stdev=np.sqrt((1/N)*(np.mean(I_squared)-I_hat**2))
confidence_interval=(I_hat-I_hat_stdev*1.96,I_hat+I_hat_stdev*1.96)
print(I_hat)
print(I_hat_stdev)
print(confidence_interval)


#%%
#Control Variates
I_estimate=[]
L_estimate=[]
for i in range(1000):
    u=unif.rvs()
    x=beta.ppf(u)
    sample_x=f(x)/beta.pdf(x)
    I_estimate.append(sample_x)
    L_estimate.append(u)

I_estimate=np.asarray(I_estimate)
L_estimate=np.asarray(L_estimate)
I_hat=np.mean(I_estimate)
L_hat=np.mean(L_estimate)
c=np.sum((I_estimate-I_hat)*(L_estimate-L_hat))/np.sum((L_estimate-L_hat)**2)
I_squared=[]
I_estimate=[]
for i in range(N):
    u=unif.rvs()
    x=beta.ppf(u)
    sample_x=f(x)/beta.pdf(x)
    controlled_sample=sample_x-c*(u-0.5)    
    I_estimate.append(controlled_sample)
    I_squared.append(controlled_sample**2)
I_estimate=np.asarray(I_estimate)
I_estimate=np.asarray(I_estimate)

I_squared=np.asarray(I_squared)
I_hat=np.mean(I_estimate)
I_hat_stdev=np.sqrt((1/N)*(np.mean(I_squared)-I_hat**2))
confidence_interval=(I_hat-I_hat_stdev*1.96,I_hat+I_hat_stdev*1.96)
print(I_hat)
print(I_hat_stdev)
print(confidence_interval)