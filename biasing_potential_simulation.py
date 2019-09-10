#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 10:32:45 2019

@author: owenmadin
"""

from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
#import yaml
from LennardJones_correlations import LennardJones
from LennardJones_2Center_correlations import LennardJones_2C
from scipy.stats import distributions
from scipy.stats import linregress
from scipy.optimize import minimize
import random as rm
from pymc3.stats import hpd
from RJMC_auxiliary_functions import *
from datetime import date
import copy
from pymbar import BAR,timeseries
import random
import sys
from RJMC_2CLJ_AUA_Q import run_RJMC


#Biasing Potential Simulation

output=[]
num_trials=1
for i in range(num_trials):
    
    compound='C2H4'
    properties=['All','three']
    temp_range=[0.55,0.95]
    n_points=20
    prior_type=['exponential','exponential']
    eps_sig_L_prior_params=[0.1,0.1,0.1]
    prior_max_values=[100,5,3,5]
    Q_prior=[0,5]
    n_iter=10**6
    swap_freq=0.0
    
    trace,logp_trace,percent_deviation_trace, attempt_matrix,acceptance_matrix,prop_sd,accept_vector,alpha_vector=run_RJMC(compound,properties,temp_range,n_points,eps_sig_L_prior_params,Q_prior,prior_type,n_iter,initial_model='AUA',swap_freq=swap_freq)
    
    avg_logP_AUA=(np.mean(logp_trace[100000:]))
    
    compound='C2H4'
    properties=['All','three']
    temp_range=[0.55,0.95]
    n_points=20
    prior_type=['exponential','exponential']
    eps_sig_L_prior_params=[0.1,0.1,0.1]
    Q_prior=[0,5]
    n_iter=10**6
    swap_freq=0.0
    prior_max_values=[100,5,3,5]
    
    trace,logp_trace,percent_deviation_trace, attempt_matrix,acceptance_matrix,prop_sd,accept_vector,alpha_vector=run_RJMC(compound,properties,temp_range,n_points,eps_sig_L_prior_params,Q_prior,prior_type,n_iter,initial_model='UA',swap_freq=swap_freq)
    
    
    avg_logP_UA=(np.mean(logp_trace[100000:]))
    
    
    logp_difference=avg_logP_UA-avg_logP_AUA
    
    compound='C2H4'
    properties=['All','three']
    temp_range=[0.55,0.95]
    n_points=20
    prior_type=['exponential','exponential']
    eps_sig_L_prior_params=[0.1,0.1,0.1]
    Q_prior=[0,5]
    n_iter=10**6
    swap_freq=0.0
    prior_max_values=[100,5,3,5]
    
    trace,logp_trace,percent_deviation_trace, attempt_matrix,acceptance_matrix,prop_sd,accept_vector,alpha_vector=run_RJMC(compound,properties,temp_range,n_points,eps_sig_L_prior_params,Q_prior,prior_type,n_iter,initial_model='AUA+Q',swap_freq=swap_freq)
    
    
    avg_logP_AUA_Q=(np.mean(logp_trace[100000:]))
    
    '''
    logp_difference=avg_logP_UA-avg_logP_AUA
    
    
    
    compound='C2H6'
    properties=['All','three']
    temp_range=[0.55,0.95]
    n_points=20
    prior_type=['gamma','gamma']
    eps_sig_L_prior_params=[40,40,40]
    Q_prior=[1,0,0.5]
    n_iter=2*10**6
    
    
    trace,logp_trace,percent_deviation_trace, attempt_matrix,acceptance_matrix,prop_sd,accept_vector,alpha_vector,thermo_data_rhoL,thermo_data_Pv,thermo_data_SurfTens,Tc_lit=run_RJMC(compound,properties,temp_range,n_points,eps_sig_L_prior_params,Q_prior,prior_type,n_iter,biasing_factor_AUA=0,return_thermo_data='True')
    '''
        
    #%%
    # POST PROCESSING
    tune_for=10000
    print('Attempted Moves')
    print(attempt_matrix)
    print('Accepted Moves')
    print(acceptance_matrix)
    prob_matrix=acceptance_matrix/attempt_matrix
    transition_matrix=np.ones((3,3))
    transition_matrix[0,1]=acceptance_matrix[0,1]/np.sum(attempt_matrix,1)[0]
    transition_matrix[0,2]=acceptance_matrix[0,2]/np.sum(attempt_matrix,1)[0]
    transition_matrix[1,0]=acceptance_matrix[1,0]/np.sum(attempt_matrix,1)[1]
    transition_matrix[1,2]=acceptance_matrix[1,2]/np.sum(attempt_matrix,1)[1]
    transition_matrix[2,1]=acceptance_matrix[2,1]/np.sum(attempt_matrix,1)[2]
    transition_matrix[2,0]=acceptance_matrix[2,0]/np.sum(attempt_matrix,1)[2]
    transition_matrix[0,0]=1-transition_matrix[0,1]-transition_matrix[0,2]
    transition_matrix[1,1]=1-transition_matrix[1,0]-transition_matrix[1,2]
    transition_matrix[2,2]=1-transition_matrix[2,0]-transition_matrix[2,1]
    print('Transition Matrix:')
    print(transition_matrix)
    trace_tuned = trace[tune_for:]
    logp_trace_tuned = logp_trace[tune_for:]
    trace_tuned[:,2:]*=10
    percent_deviation_trace_tuned = percent_deviation_trace[tune_for:]
    model_params = trace_tuned[0,:]
    
    #[t0,g,Neff_max]=timeseries.detectEquilibration(logp_trace_tuned,nskip=100)
    
    #logp_trace_equil=logp_trace_tuned[t0:]
    trace_equil = trace_tuned
    #percent_deviation_trace_equil = percent_deviation_trace_tuned[t0:]
    
    print(len(trace_equil))
    
    fname=compound+'likelihood_data_amount_10'+'_'+properties[0]+'_'+str(n_points)+'_'+str(n_iter)+'_'+str(date.today())
    
    lit_params,lit_devs=import_literature_values(properties[1],compound)
    #new_lit_devs=computePercentDeviations(thermo_data_rhoL[:,0],thermo_data_Pv[:,0],thermo_data_SurfTens[:,0],lit_devs,thermo_data_rhoL[:,1],thermo_data_Pv[:,1],thermo_data_SurfTens[:,1],Tc_lit[0],rhol_hat_models,Psat_hat_models,SurfTens_hat_models,T_c_hat_models)
    
    #%%
    #new_lit_devs=recompute_lit_percent_devs(lit_params,computePercentDeviations,thermo_data_rhoL[:,0],thermo_data_Pv[:,0],thermo_data_SurfTens[:,0],lit_devs,thermo_data_rhoL[:,1],thermo_data_Pv[:,1],thermo_data_SurfTens[:,1],Tc_lit[0],rhol_hat_models,Psat_hat_models,SurfTens_hat_models,T_c_hat_models,compound_2CLJ)
    #pareto_point,pareto_point_values=findParetoPoints(percent_deviation_trace_tuned,trace_tuned,0)
    
    
    '''
    max_ap = np.zeros(np.size(model_params))
    map_CI = np.zeros((np.size(model_params),2))
    for i in range(np.size(model_params)):
        bins,values=np.histogram(trace_tuned[:,i],bins=100,density=True)
        max_ap[i]=(values[np.argmax(bins)+1]+values[np.argmax(bins)])/2
        map_CI[i]=hpd(trace_tuned[:,i],alpha=0.05)
        plt.hist(trace_tuned[:,i],bins=100,label='Sampled Posterior',density='True'),plt.axvline(x=map_CI[i][0],color='red',label='HPD 95% CI',ls='--'),plt.axvline(x=map_CI[i][1],color='red',ls='--'),plt.axvline(x=max_ap[i],color='orange',lw=1,label='MAP Estimate')
        plt.axvline(x=initial_values[i],color='magenta',label='Literature/Initial Value')
        plt.legend()
        plt.show()
    max_ap[0]=np.floor(max_ap[0])
    plotPercentDeviations(percent_deviation_trace_tuned,pareto_point,'MCMC Points','Pareto Point')
    
    
    
    
    plotDeviationHistogram(percent_deviation_trace_tuned,pareto_point)
    '''
    # Converts the array with number of model parameters into an array with the number of times there was 1 parameter or 2 parameters
    model_count = np.array([len(trace_equil[trace_equil[:,0]==0]),len(trace_equil[trace_equil[:,0]==1]),len(trace_equil[trace_equil[:,0]==2])])
    
    
    prob_0 = 1.*model_count[0]/(len(trace_equil))
    print('Percent that  model 0 is sampled: '+str(prob_0 * 100.)) #The percent that use 1 parameter model
    
    prob_1 = 1.*model_count[1]/(len(trace_equil))
    print('Percent that model 1 is sampled: '+str(prob_1 * 100.)) #The percent that use two center UA LJ
    
    prob_2 = 1.*model_count[2]/(len(trace_equil))
    print('Percent that model 2 is sampled: '+str(prob_2 * 100.)) #The percent that use two center UA LJ
    
    prob=[prob_0,prob_1,prob_2]
    '''
    if logp_difference >= 0:
        adjusted_prob_0 = prob_0*np.exp(logp_difference)
        adjusted_prob_1 = 1- adjusted_prob_0
    elif logp_difference <= 0:
        adjusted_prob_2 = prob_1*np.exp(logp_difference)
        adjusted_prob_0 = 1- adjusted_prob_2
    '''
    output.append([prob_0,prob_1])
print(output)
            