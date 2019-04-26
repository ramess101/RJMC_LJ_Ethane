#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 12:04:57 2018

MCMC/RJMC Toolbox: A class of functions that implement RJMC/MCMC algorithms, tailored towards the use of RJMC


@author: owenmadin
"""


from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from scipy.stats import distributions
from scipy.stats import linregress
from scipy.optimize import minimize
import random as rm
from pymc3.stats import hpd


#%%
# Simplify notation
dnorm = distributions.norm.logpdf
dgamma = distributions.gamma.logpdf
duni = distributions.uniform.logpdf

rnorm = np.random.normal
runif = np.random.rand

norm=distributions.norm.pdf
unif=distributions.uniform.pdf


'''
properties = 'rhol'

def calc_posterior(model,eps,sig,Q):

    logp = 0
#    print(eps,sig)
    # Using noninformative priors
    logp += duni(sig, 0.2, 0.5)
    logp += duni(eps, 100,200) 
    
    if model == 0:
        Q=0
    
    
    if model == 1:
        logp+=duni(Q,0,2)
    
    # OCM: no reason to use anything but uniform priors at this point.  Could probably narrow the prior ranges a little bit to improve acceptance,
    #But Rich is rightly being conservative here especially since evaluations are cheap.
    
#    print(eps,sig)
    #rhol_hat_fake = rhol_hat_models(T_lin,model,eps,sig)
    rhol_hat = rhol_hat_models(T_rhol_data,model,eps,sig,Q) #[kg/m3]
    Psat_hat = Psat_hat_models(T_Psat_data,model,eps,sig,Q) #[kPa]        
 
    # Data likelihood
    if properties == 'rhol':
        logp += sum(dnorm(rhol_data,rhol_hat,t_rhol**-2.))
        #logp += sum(dnorm(rhol_data,rhol_hat,t_rhol**-2.))
    elif properties == 'Psat':
        logp += sum(dnorm(Psat_data,Psat_hat,t_Psat**-2.))
    elif properties == 'Multi':
        logp += sum(dnorm(rhol_data,rhol_hat,t_rhol**-2.))
        logp += sum(dnorm(Psat_data,Psat_hat,t_Psat**-2.))
    return logp
'''
def RJMC_outerloop(calc_posterior,n_iterations,initial_values,initial_sd,n_models,swap_freq,tune_freq,tune_for,jacobian,transition_function,AUA_opt_params,AUA_Q_opt_params):
    
    
    #INITIAL SETUP FOR MC LOOP
    #-----------------------------------------------------------------------------------------#
    
    n_params = len(initial_values) #One column is the model number
    accept_vector=np.zeros((n_iterations))
    prop_sd=initial_sd
    
    #Initialize matrices to count number of moves of each type
    attempt_matrix=np.zeros((n_models,n_models))
    acceptance_matrix=np.zeros((n_models,n_models))
    
    
    # Initialize trace for parameters
    trace = np.zeros((n_iterations+1, n_params)) #n_iterations + 1 to account for guess
    logp_trace = np.zeros(n_iterations+1)
    # Set initial values
    trace[0] = initial_values
    
    # Calculate joint posterior for initial values
    current_log_prob = calc_posterior(*trace[0])
    
    logp_trace[0] = current_log_prob
    current_params=trace[0].copy()
    record_acceptance='False'
    #----------------------------------------------------------------------------------------#
    
    #OUTER MCMC LOOP
    
    for i in range(n_iterations):
        if not i%5000: print('Iteration '+str(i))
        
        
        # Grab current parameter values
        current_params = trace[i].copy()
        current_model = int(current_params[0])
        current_log_prob = logp_trace[i].copy()
        
        if i >= tune_for:
            record_acceptance='True'
        
        new_params, new_log_prob, attempt_matrix,acceptance_matrix,acceptance = RJMC_Moves(current_params,current_model,current_log_prob,n_models,swap_freq,n_params,prop_sd,attempt_matrix,acceptance_matrix,jacobian,transition_function,record_acceptance,AUA_opt_params,AUA_Q_opt_params)
        
        if acceptance == 'True':
            accept_vector[i]=1
        logp_trace[i+1] = new_log_prob
        trace[i+1] = new_params
        
        if (not (i+1) % tune_freq) and (i < tune_for):
        
            print('Tuning on step %1.1i' %i)
            #print(np.sum(accept_vector[i-tune_freq:]))
            acceptance_rate = np.sum(accept_vector)/i           
            print(acceptance_rate)
            for m in range (n_params-1):
                if acceptance_rate<0.2:
                    prop_sd[m+1] *= 0.9
                    print('Yes')
                elif acceptance_rate>0.5:
                    prop_sd[m+1] *= 1.1
                    print('No')         
           
            
    return trace,logp_trace, attempt_matrix,acceptance_matrix,prop_sd,accept_vector

def RJMC_Moves(current_params,current_model,current_log_prob,n_models,swap_freq,n_params,prop_sd,attempt_matrix,acceptance_matrix,jacobian,transition_function,record_acceptance,AUA_opt_params,AUA_Q_opt_params):
    
    params = current_params.copy()# This approach updates previous param values
    #Grab a copy of the current params to work with
    #current_log_prob_copy=copy.deepcopy(current_log_prob)
    
    #Roll a dice to decide what kind of move will be suggested
    mov_ran=np.random.random()
    
    if mov_ran <= swap_freq:
        #mu=0.015
        
        params,rjmc_jacobian,proposed_log_prob,proposed_model,w,lamda,transition_function=model_proposal(current_model,n_models,n_params,params,jacobian,transition_function,AUA_opt_params,AUA_Q_opt_params)
        
        alpha = (proposed_log_prob - current_log_prob) + np.log(rjmc_jacobian) + np.log(transition_function)
        
        acceptance=accept_reject(alpha)
        
        if acceptance =='True':
            new_log_prob=proposed_log_prob
            new_params=params
            if record_acceptance == 'True':
                acceptance_matrix[current_model,proposed_model]+=1
                attempt_matrix[current_model,proposed_model]+=1
        elif acceptance == 'False':
            new_params=current_params
            new_log_prob=current_log_prob
            if record_acceptance == 'True':
                attempt_matrix[current_model,proposed_model]+=1
        '''
        move_type = 'Swap'
    else: 
        move_type = 'Trad'
    
        
    if move_type == 'Swap':
        '''
    else:
        params,proposed_log_prob=parameter_proposal(params,n_params,prop_sd)    
        
        alpha = (proposed_log_prob - current_log_prob)
    
        acceptance=accept_reject(alpha)
                    
    
        if acceptance =='True':
            new_log_prob=proposed_log_prob
            new_params=params
            if record_acceptance == 'True':
                 acceptance_matrix[current_model,current_model]+=1
                 attempt_matrix[current_model,current_model]+=1
        elif acceptance == 'False':
             new_params=current_params
             new_log_prob=current_log_prob
             if record_acceptance == 'True':
                attempt_matrix[current_model,current_model]+=1
    
                   
    return new_params,new_log_prob,attempt_matrix,acceptance_matrix,acceptance


def accept_reject(alpha):    
    urv=runif()
    if np.log(urv) < alpha:  
        acceptance='True'
    else: 
        acceptance='False'
    return acceptance
        
def model_proposal(current_model,n_models,n_params,params,jacobian,transition_function,AUA_opt_params,AUA_Q_opt_params):
    proposed_model=current_model
    while proposed_model==current_model:
        proposed_model=int(np.floor(np.random.random()*n_models))
        
    lamda=2    
    params[0] = proposed_model
    if proposed_model==1:
        
        params[1] = (AUA_Q_opt_params[0]/AUA_opt_params[0])*params[1]
        params[2] = (AUA_Q_opt_params[1]/AUA_opt_params[1])*params[2]
        w=runif()
        
        #THIS IS IMPORTANT needs to be different depending on which direction
        
        
        params[3] = -(1/lamda)*np.log(w)

    if proposed_model==0:
        params[1] = (AUA_opt_params[0]/AUA_Q_opt_params[0])*params[1]
        params[2] = (AUA_opt_params[1]/AUA_Q_opt_params[1])*params[2]
        w=np.exp(-lamda*params[3])
        params[3]=0
        

    proposed_log_prob=calc_posterior(*params)
    jacobian =  jacobian(n_models,n_params,w,lamda,AUA_opt_params,AUA_Q_opt_params)
    rjmc_jacobian=jacobian[current_model,proposed_model]
    transition_function=transition_function(n_models,w)
    transition_function=transition_function[current_model,proposed_model]
    return params,rjmc_jacobian,proposed_log_prob,proposed_model,w,lamda,transition_function


def parameter_proposal(params,n_params,prop_sd):
    proposed_param=int(np.ceil(np.random.random()*(n_params-1)))
    params[proposed_param] = rnorm(params[proposed_param], prop_sd[proposed_param])
    proposed_log_prob=calc_posterior(*params)
    if params[0]==0:
        params[3]=0
    return params, proposed_log_prob



def mcmc_prior_proposal(n_models,calc_posterior,guess_params,guess_sd):
    swap_freq=0.0
    n_iter=50000
    tune_freq=100
    tune_for=10000
    
    for i in range(n_models):
        initial_values=guess_params[i,:]
        initial_sd=guess_sd[i,:]
        trace,logp_trace,attempt_matrix,acceptance_matrix,prop_sd,accept_vector = RJMC_outerloop(calc_posterior,n_iter,initial_values,initial_sd,n_models,swap_freq,tune_freq,tune_for,1,1,1,1)
        trace_tuned = trace[tune_for:]
        max_ap=np.empty(np.size(trace_tuned,1))
        map_CI=np.zeros((np.size(trace_tuned,1),2))
        parameter_prior_proposal=np.empty((n_models,np.size(trace_tuned,1),2))

        for j in range(np.size(trace_tuned,2)):
            bins,values=np.histogram(trace_tuned[:,j],bins=100)
            max_ap[j]=(values[np.argmax(bins)+1]+values[np.argmax(bins)])/2
            map_CI[j]=hpd(trace_tuned[:,i],alpha=0.05)
            sigma_hat=map_CI[j,1]-map_CI[j,0]/(2*1.96)
            parameter_prior_proposal[j,i]=[max_ap,sigma_hat*1.5]
            support=np.linspace(np.min(trace_tuned[:,j]),np.max(trace_tuned[:,j]),100)
            plt.hist(trace_tuned[:,j],density=True)
            plt.plot(support,norm(support,*parameter_prior_proposal))
    return parameter_prior_proposal