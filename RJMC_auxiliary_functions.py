#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 11:08:09 2019

@author: owenmadin
"""

from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from LennardJones_correlations import LennardJones
from LennardJones_2Center_correlations import LennardJones_2C
from scipy.stats import distributions
from scipy.stats import linregress
from scipy.optimize import minimize
import random as rm
from pymc3.stats import hpd


# Here we have chosen ethane as the test case
compound="ethane"
fname = compound+".yaml"

# Load property values for compound

with open(fname) as yfile:
    yfile = yaml.load(yfile)

eps_lit_LJ = yfile["force_field_params"]["eps_lit_LJ"] #[K]
sig_lit_LJ = yfile["force_field_params"]["sig_lit_LJ"] #[nm]
eps_lit_UA = yfile["force_field_params"]["eps_lit_UA"] #[K]
sig_lit_UA = yfile["force_field_params"]["sig_lit_UA"] #[nm]
Lbond_lit_UA = yfile["force_field_params"]["Lbond_lit_UA"] #[nm]
eps_lit_AUA = yfile["force_field_params"]["eps_lit_AUA"] #[K]
sig_lit_AUA = yfile["force_field_params"]["sig_lit_AUA"] #[nm]
Lbond_lit_AUA = yfile["force_field_params"]["Lbond_lit_AUA"] #[nm]
eps_lit2_AUA = yfile["force_field_params"]["eps_lit2_AUA"] #[K]
sig_lit2_AUA = yfile["force_field_params"]["sig_lit2_AUA"] #[nm]
Lbond_lit2_AUA = yfile["force_field_params"]["Lbond_lit2_AUA"] #[nm]
Q_lit2_AUA = yfile["force_field_params"]["Q_lit2_AUA"] #[DAng]
eps_lit3_AUA = yfile["force_field_params"]["eps_lit3_AUA"] #[K]
sig_lit3_AUA = yfile["force_field_params"]["sig_lit3_AUA"] #[nm]
Lbond_lit3_AUA = yfile["force_field_params"]["Lbond_lit3_AUA"] #[nm]
Q_lit3_AUA = yfile["force_field_params"]["Q_lit3_AUA"] #[DAng]
Tc_RP = yfile["physical_constants"]["T_c"] #[K]
rhoc_RP = yfile["physical_constants"]["rho_c"] #[kg/m3]
M_w = yfile["physical_constants"]["M_w"] #[gm/mol]

# Substantiate LennardJones class
Ethane_LJ = LennardJones(M_w)
Ethane_2CLJ = LennardJones_2C(M_w)

# Epsilon and sigma can be obtained from the critical constants
eps_Tc = Ethane_LJ.calc_eps_Tc(Tc_RP) #[K]
sig_rhoc = Ethane_LJ.calc_sig_rhoc(rhoc_RP) #[nm]

# Create functions that return properties for a given model, eps, sig

def rhol_hat_models(Temp,model,eps,sig,L,Q):
    
    if model == 0: #Two center AUA LJ
    
        rhol_hat = Ethane_2CLJ.rhol_hat_2CLJQ(Temp,eps,sig,L,0) 
        
    elif model == 1: #Two center AUA LJ+Q
    
        rhol_hat = Ethane_2CLJ.rhol_hat_2CLJQ(Temp,eps,sig,L,Q) 
        
    return rhol_hat #[kg/m3]       
  
def Psat_hat_models(Temp,model,eps,sig,L,Q):
    
    if model == 0: #Two center AUA LJ
    
        Psat_hat = Ethane_2CLJ.Psat_hat_2CLJQ(Temp,eps,sig,L,0) 
        
    elif model == 1: #Two center AUA LJ+Q
    
        Psat_hat = Ethane_2CLJ.Psat_hat_2CLJQ(Temp,eps,sig,L,Q) 
        
    return Psat_hat #[kPa]       

'''
# Load REFPROP data from file so that user does not need REFPROP
data = np.loadtxt('TRC_deltaHv.txt')
T_deltaHv = data[:,0] #[K]
RP_deltaHv = data[:,1] #[kJ/mol]

data = np.loadtxt('TRC_data_rhoL.txt')
T_rhol_data = data[:,0] #[K]
rhol_data = data[:,1] #[kJ/mol]

data = np.loadtxt('TRC_data_Pv.txt')
T_Psat_data = data[:,0] #[K]
Psat_data = data[:,1] #[kJ/mol]

# Limit temperature range to that which is typical of ITIC MD simulations

T_min = 167.9
T_max = 290.1

rhol_data = rhol_data[T_rhol_data>T_min]
T_rhol_data = T_rhol_data[T_rhol_data>T_min]
rhol_data = rhol_data[T_rhol_data<T_max]
T_rhol_data = T_rhol_data[T_rhol_data<T_max]

Psat_data = Psat_data[T_Psat_data>T_min]
T_Psat_data = T_Psat_data[T_Psat_data>T_min]
Psat_data = Psat_data[T_Psat_data<T_max]
T_Psat_data = T_Psat_data[T_Psat_data<T_max]
'''
def computePercentDeviations(temp_values_rhol,temp_values_psat,parameter_values,rhol_data,psat_data):
    
    
    rhol_model=rhol_hat_models(temp_values_rhol,*parameter_values)
    psat_model=Psat_hat_models(temp_values_psat,*parameter_values)
    
    rhol_deviation_vector=((rhol_data-rhol_model)/rhol_data)**2
    psat_deviation_vector=((psat_data-psat_model)/psat_data)**2

    rhol_mean_relative_deviation=np.sqrt(np.sum(rhol_deviation_vector)/np.size(rhol_deviation_vector))*100
    psat_mean_relative_deviation=np.sqrt(np.sum(psat_deviation_vector)/np.size(psat_deviation_vector))*100
    
    return rhol_mean_relative_deviation, psat_mean_relative_deviation
    
        
def plotPercentDeviations(percent_deviation_trace,max_apd,label1,label2):
    
    plt.scatter(percent_deviation_trace[:,0],percent_deviation_trace[:,1],alpha=0.5,marker='x',label=label1)
    plt.scatter(max_apd[0],max_apd[1],alpha=1,marker='x',color='r',label=label2)
    plt.scatter(percent_deviation_trace[0,0],percent_deviation_trace[0,1],alpha=1,marker='x',color='orange',label='Literature')
    plt.xlabel(r'% Deviation, $\rho_l$')
    plt.ylabel(r'% Deviation, $P_{sat}$')
    plt.gca().set_xlim(left=0)
    plt.gca().set_ylim(bottom=0)
    plt.legend()
    plt.show()
    return

def plotDeviationHistogram(percent_deviation_trace,pareto_point):
    plt.gca().set_xlim(left=0)
    plt.gca().set_ylim(bottom=0)
    plt.hist2d(percent_deviation_trace[:,0],percent_deviation_trace[:,1],bins=100,range=[[0,np.max(percent_deviation_trace[:,0])],[0,np.max(percent_deviation_trace[:,1])]])
    plt.scatter(pareto_point[0],pareto_point[1],color='r',marker='.',alpha=0.5)
    plt.scatter(percent_deviation_trace[0,0],percent_deviation_trace[0,1],alpha=1,marker='x',color='orange',label='Literature')
    plt.xlabel(r'% Deviation, $\rho_l$')
    plt.ylabel(r'% Deviation, $P_{sat}$')

    plt.show()
    
    
    plt.hist(percent_deviation_trace[:,0],bins=100,density=True)
    plt.xlabel(r'% Deviation, $\rho_l$')
    plt.ylabel('Probability Density')
    plt.show()

    plt.hist(percent_deviation_trace[:,1],bins=100,density=True)
    plt.xlabel(r'% Deviation, $P_{sat}$')
    plt.ylabel('Probability Density')
    plt.show()
    
    return

def findParetoPoints(percent_deviation_trace,trace):
    total_percent_dev=np.sum(percent_deviation_trace,1)
    pareto_point=percent_deviation_trace[np.argmin(total_percent_dev)]
    pareto_point_values=trace[np.argmin(total_percent_dev)]
    return pareto_point,pareto_point_values
