#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 19:34:17 2018

Implementation of RJMC between AUA and AUA-Q models.

    
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
from RJMC_auxiliary_functions import *
from datetime import date
import copy
from pymbar import BAR,timeseries
import random
import sys
#import multiprocessing as mp


# Here we have chosen ethane as the test case


compound=sys.argv[1]
ff_params_ref,Tc_lit,M_w,thermo_data,NIST_bondlength=parse_data_ffs(compound)
#Retrieve force field literature values, constants, and thermo data


T_min = float(sys.argv[2])*Tc_lit[0]
T_max = float(sys.argv[3])*Tc_lit[0]
n_points=int(sys.argv[4])

#Select temperature range of data points to select, and how many temperatures within that range to use data at. 


thermo_data=filter_thermo_data(thermo_data,T_min,T_max,n_points)
#Filter data to selected conditions.


uncertainties=calculate_uncertainties(thermo_data,Tc_lit[0])
#Calculate uncertainties for each data point, based on combination of experimental uncertainty and correlation uncertainty

thermo_data_rhoL=np.asarray(thermo_data['rhoL'])
thermo_data_Pv=np.asarray(thermo_data['Pv'])
thermo_data_SurfTens=np.asarray(thermo_data['SurfTens'])
#Convert dictionaries to numpy arrays


# Substantiate LennardJones class
compound_2CLJ = LennardJones_2C(M_w)


'''
# Epsilon and sigma can be obtained from the critical constants
#eps_Tc = Ethane_LJ.calc_eps_Tc(Tc_RP) #[K]
#sig_rhoc = Ethane_LJ.calc_sig_rhoc(rhoc_RP) #[nm]


# Set percent uncertainty in each property
# These values are to represent the simulation uncertainty more than the experimental uncertainty
# Also, the transiton matrix for eps and sig for each model are tuned to this rhol uncertainty.
# I.e. the optimal "lit" values agree well with a 3% uncertainty in rhol. This improved the RJMC model swap acceptance.
pu_rhol = 3
pu_Psat = 5

# I decided to include the same error model I am using for Mie lambda-6
# For pu_rhol_low = 0.3 and pu_rhol_high = 0.5 AUA is 100%
# For pu_rhol_low = 1 and pu_rhol_high = 3 LJ 16%, UA 22%, AUA 62%
#pu_rhol_low = 1
#T_rhol_switch = 230
#pu_rhol_high = 3
#
#pu_Psat_low = 5
#T_Psat_switch = 180
#pu_Psat_high = 3
#
## Piecewise function to represent the uncertainty in rhol and Psat
#pu_rhol = np.piecewise(T_rhol_data,[T_rhol_data<T_rhol_switch,T_rhol_data>=T_rhol_switch],[pu_rhol_low,lambda x:np.poly1d(np.polyfit([T_rhol_switch,T_max],[pu_rhol_low,pu_rhol_high],1))(x)])
#pu_Psat = np.piecewise(T_Psat_data,[T_Psat_data<T_Psat_switch,T_Psat_data>=T_Psat_switch],[lambda x:np.poly1d(np.polyfit([T_min,T_Psat_switch],[pu_Psat_low,pu_Psat_high],1))(x),pu_Psat_high])
  
# Calculate the absolute uncertainty
#u_rhol = rhol_data*pu_rhol/100.
#u_Psat = Psat_data*pu_Psat/100.
'''

   
# Calculate the estimated standard deviation
sd_rhol = uncertainties['rhoL']/2.
sd_Psat = uncertainties['Pv']/2.
sd_SurfTens = uncertainties['SurfTens']/2

# Calculate the precision in each property
t_rhol = np.sqrt(1./sd_rhol)
t_Psat = np.sqrt(1./sd_Psat)                                   
t_SurfTens = np.sqrt(1./sd_SurfTens)                                   

# Initial values for the Markov Chain

guess_0 = [0,*ff_params_ref[1]]
guess_1 = [1,*ff_params_ref[0]]
guess_2 = [2,*ff_params_ref[2]]
# Create initial starting points based on previous optimization data

guess_2[3] = NIST_bondlength
#Modify Bond length for UA model to experimental value
#guess_2 = [1,eps_lit3_AUA,sig_lit3_AUA,Lbond_lit3_AUA,Q_lit3_AUA] 

#%%
# Simplify notation ( we will use these functions to create priors and draw RVs as needed )
dnorm = distributions.norm.logpdf
dgamma = distributions.gamma.logpdf
duni = distributions.uniform.logpdf
dlogit = distributions.logistic.logpdf

gammarvs=distributions.gamma.rvs
logitrvs=distributions.logistic.rvs
uniformrvs=distributions.uniform.rvs

rnorm = np.random.normal
runif = np.random.rand

norm=distributions.norm.pdf
unif=distributions.uniform.pdf

#Select number of properties and which properties (current options are 'rhol','Psat', 'rhol+Psat','All')

properties = 'rhol+Psat'
number_criteria = 'two'


prior_range=float(sys.argv[5])

#Uniform Priors (creating uniform priors based on optimization values)
eps_prior=[ff_params_ref[1][0]*(1-prior_range),ff_params_ref[1][0]*(1+prior_range)]
sig_prior=[ff_params_ref[1][1]*(1-prior_range),ff_params_ref[1][1]*(1+prior_range)]
L_prior=[ff_params_ref[1][2]*(1-prior_range),ff_params_ref[1][2]*(1+prior_range)]



#Logistic priors (creating logistic priors based to optimization values)

'''
shape_divide=float(sys.argv[5])

eps_prior=[ff_params_ref[1][0],ff_params_ref[1][0]/shape_divide]
sig_prior=[ff_params_ref[1][1],ff_params_ref[1][1]/shape_divide]
L_prior=[ff_params_ref[1][2],ff_params_ref[1][2]/shape_divide]
'''
#Q priors
#Can use uniform or gamma prior
#Uniform
#Q_prior=[0,0.3]
#Gamma
Q_prior=[1,0,0.5]

def calc_posterior(model,eps,sig,L,Q):

    logp = 0
    logp += duni(sig, *sig_prior)
    logp += duni(eps, *eps_prior)  
    #Create priors for parameters common to all models     
    if model == 2:
        Q=0
        #Ensure Q=0 for UA model
    
    elif model == 0:
        Q=0
        logp+=duni(L,*L_prior)
        #Add prior over L for AUA model
    
    elif model == 1:
        logp+=dgamma(Q,*Q_prior)
        logp+=duni(L,*L_prior)

        #Add priors for Q and L for AUA+Q model
        
    rhol_hat = rhol_hat_models(compound_2CLJ,thermo_data_rhoL[:,0],model,eps,sig,L,Q) #[kg/m3]
    Psat_hat = Psat_hat_models(compound_2CLJ,thermo_data_Pv[:,0],model,eps,sig,L,Q) #[kPa]   
    SurfTens_hat = SurfTens_hat_models(compound_2CLJ,thermo_data_SurfTens[:,0],model,eps,sig,L,Q)     
    #Compute properties at temperatures from experimental data
    
    
    # Data likelihood: Compute likelihood based on gaussian penalty function
    if properties == 'rhol':
        logp += sum(dnorm(thermo_data_rhoL[:,1],rhol_hat,t_rhol**-2.))
        #logp += sum(dnorm(rhol_data,rhol_hat,t_rhol**-2.))
    elif properties == 'Psat':
        logp += sum(dnorm(thermo_data_Pv[:,1],Psat_hat,t_Psat**-2.))
    elif properties == 'rhol+Psat':
        logp += sum(dnorm(thermo_data_rhoL[:,1],rhol_hat,t_rhol**-2.))
        logp += sum(dnorm(thermo_data_Pv[:,1],Psat_hat,t_Psat**-2.))
    elif properties == 'All':
         logp += sum(dnorm(thermo_data_rhoL[:,1],rhol_hat,t_rhol**-2.))
         logp += sum(dnorm(thermo_data_Pv[:,1],Psat_hat,t_Psat**-2.))           
         logp += sum(dnorm(thermo_data_SurfTens[:,1],SurfTens_hat,t_SurfTens**-2))
    return logp
    #return rhol_hat
    
    
def jacobian(n_models,n_params,w,lamda,opt_params_AUA,opt_params_AUA_Q,opt_params_2CLJ):
    jacobian=np.ones((n_models,n_models))
    
    
    
    #Optimum Matching for UA --> AUA
    #jacobian[0,1]=(1/(lamda*w))*(opt_params_AUA_Q[0]*opt_params_AUA_Q[1]*opt_params_AUA_Q[2])/(opt_params_AUA[0]*opt_params_AUA[1]*opt_params_AUA[2])
    #jacobian[1,0]=lamda*(opt_params_AUA[0]*opt_params_AUA[1]*opt_params_AUA[2])/(opt_params_AUA_Q[0]*opt_params_AUA_Q[1]*opt_params_AUA_Q[2])
    jacobian[0,2]=(opt_params_2CLJ[0]*opt_params_2CLJ[1]*opt_params_2CLJ[2])/(opt_params_AUA[0]*opt_params_AUA[1]*opt_params_AUA[2])
    jacobian[2,0]=(opt_params_AUA[0]*opt_params_AUA[1]*opt_params_AUA[2])/(opt_params_2CLJ[0]*opt_params_2CLJ[1]*opt_params_2CLJ[2])
    #Direct transfer for AUA->AUA+Q 
    jacobian[0,1]=1/(lamda*w)
    jacobian[1,0]=w*lamda
    
    
    #jacobian[0,1]=(1/(lamda*w))*(AUA_Q_opt_params[0]*AUA_Q_opt_params[1])/(AUA_opt_params[0]*AUA_opt_params[1])
    #jacobian[1,0]=w*lamda*(AUA_opt_params[0]*AUA_opt_params[1])/(AUA_Q_opt_params[0]*AUA_Q_opt_params[1])
    #jacobian[0,1]=1/(lamda*w)
    #jacobian[1,0]=w*lamda
    
    return jacobian
    

def transition_function(n_models,w):
    transition_function=np.ones((n_models,n_models))
    g_0_1=unif(w,0,1)
    g_1_0=1
    g_0_2=1
    g_2_0=1
    #These are proposal distributions for "new" variables (that exist in one model but not the other).  They have been cleverly chosen to all equal 1
    
    
    q_0_1=1/2
    q_1_0=1
    q_0_2=1/2
    q_2_0=1
    #These are probabilities of proposing a model from one model to another.  
    #The probability is half for moves originating in AUA because they can move either to UA or AUA+Q. We disallow moves between UA and AUA+Q directly 
    
    #Note that this is really times swap_freq but that term always cancels.
    
    
    transition_function[0,1]=g_1_0*q_1_0/(g_0_1*q_0_1)
    transition_function[1,0]=g_0_1*q_0_1/(g_1_0*q_1_0)
    transition_function[0,2]=g_2_0*q_2_0/(g_0_2*q_0_2)
    transition_function[2,0]=g_0_2*q_0_2/(g_2_0*q_2_0)
    #Transition functions enumerated for each 
    
    
    return transition_function


def gen_Tmatrix():
    ''' Generate Transition matrices based on the optimal eps, sig, Q for different models'''
    
    #Currently this is not used for moves between AUA and AUA+Q, because it doesn't seem to help.  Still used for UA and AUA moves
    
    obj_AUA = lambda eps_sig_Q: -calc_posterior(0,eps_sig_Q[0],eps_sig_Q[1],eps_sig_Q[2],eps_sig_Q[3])
    obj_AUA_Q = lambda eps_sig_Q: -calc_posterior(1,eps_sig_Q[0],eps_sig_Q[1],eps_sig_Q[2],eps_sig_Q[3])
    obj_2CLJ = lambda eps_sig_Q: -calc_posterior(2,eps_sig_Q[0],eps_sig_Q[1],eps_sig_Q[2],eps_sig_Q[3])
    
    
    guess_AUA = [guess_0[1],guess_0[2],guess_0[3],guess_0[4]]
    guess_AUA_Q = [guess_1[1],guess_1[2],guess_1[3],guess_1[4]]
    guess_2CLJ = [guess_2[1],guess_2[2],guess_2[3],guess_2[4]]
    
    # Make sure bounds are in a reasonable range so that models behave properly
    bnd_AUA = ((0.85*guess_0[1],guess_0[1]*1.15),(0.90*guess_0[2],guess_0[2]*1.1),(0.90*guess_0[3],guess_0[3]*1.1),(0.90*guess_0[4],guess_0[4]*1.1))
    bnd_AUA_Q = ((0.85*guess_1[1],guess_1[1]*1.15),(0.9*guess_1[2],guess_1[2]*1.1),(0.9*guess_1[3],guess_1[3]*1.1),(0.90*guess_1[4],guess_1[4]*1.1))
    bnd_2CLJ = ((0.85*guess_2[1],guess_2[1]*1.15),(0.9*guess_2[2],guess_2[2]*1.1),(1*guess_2[3],guess_2[3]*1),(0.90*guess_2[4],guess_2[4]*1.1))
    #Help debug
#    print(bnd_LJ)
#    print(bnd_UA)
#    print(bnd_AUA)
    
    
    opt_AUA = minimize(obj_AUA,guess_AUA,bounds=bnd_AUA)
    opt_AUA_Q = minimize(obj_AUA_Q,guess_AUA_Q,bounds=bnd_AUA_Q)
    opt_2CLJ = minimize(obj_2CLJ,guess_2CLJ,bounds=bnd_2CLJ)
    #Help debug
#    print(opt_LJ)
#    print(opt_UA)
#    print(opt_AUA)
        
    opt_params_AUA = opt_AUA.x[0],opt_AUA.x[1],opt_AUA.x[2],opt_AUA.x[3]
    opt_params_AUA_Q = opt_AUA_Q.x[0],opt_AUA_Q.x[1],opt_AUA_Q.x[2],opt_AUA_Q.x[3]
    opt_params_2CLJ = opt_2CLJ.x[0],opt_2CLJ.x[1],opt_2CLJ.x[2],opt_2CLJ.x[3]
        
    return opt_params_AUA, opt_params_AUA_Q, opt_params_2CLJ

opt_params_AUA,opt_params_AUA_Q, opt_params_2CLJ = gen_Tmatrix()

#%%

#The fraction of times a model swap is suggested as the move, rather than an intra-model move

def RJMC_outerloop(calc_posterior,n_iterations,initial_values,initial_sd,n_models,swap_freq,tune_freq,tune_for,jacobian,transition_function,opt_params_AUA,opt_params_AUA_Q,opt_params_2CLJ):
    
    
    #INITIAL SETUP FOR MC LOOP
    #-----------------------------------------------------------------------------------------#
    
    n_params = len(initial_values) #One column is the model number
    accept_vector=np.zeros((n_iterations))
    prop_sd=initial_sd
    
    #Initialize matrices to count number of moves of each type
    attempt_matrix=np.zeros((n_models,n_models))
    acceptance_matrix=np.zeros((n_models,n_models))
    
    
    alpha_vector_01=[]
    alpha_vector_10=[]
    alpha_vector_02=[]
    alpha_vector_20=[]
    
    # Initialize trace for parameters
    trace = np.zeros((n_iterations+1, n_params)) #n_iterations + 1 to account for guess
    logp_trace = np.zeros(n_iterations+1)
    percent_deviation_trace = np.zeros((n_iterations+1,4))
    # Set initial values
    trace[0] = initial_values
    
    # Calculate joint posterior for initial values
    current_log_prob = calc_posterior(*trace[0])
    
    logp_trace[0] = current_log_prob
    percent_deviation_trace[0]=computePercentDeviations(compound_2CLJ,thermo_data_rhoL[:,0],thermo_data_Pv[:,0],thermo_data_SurfTens[:,0],initial_values,thermo_data_rhoL[:,1],thermo_data_Pv[:,1],thermo_data_SurfTens[:,1],Tc_lit[0],rhol_hat_models,Psat_hat_models,SurfTens_hat_models,T_c_hat_models)
    current_params=trace[0].copy()
    record_acceptance='False'
    #----------------------------------------------------------------------------------------#
    
    #OUTER MCMC LOOP
    
    for i in range(n_iterations):
        if not i%50000: print('Iteration '+str(i))
        
        
        # Grab current parameter values
        current_params = trace[i].copy()
        current_model = int(current_params[0])
        current_log_prob = logp_trace[i].copy()
        
        if i >= tune_for:
            record_acceptance='True'
        
        new_params, new_log_prob, attempt_matrix,acceptance_matrix,acceptance,alpha,model_swap_attempt,model_swap_type = RJMC_Moves(current_params,current_model,current_log_prob,n_models,swap_freq,n_params,prop_sd,attempt_matrix,acceptance_matrix,jacobian,transition_function,record_acceptance,opt_params_AUA,opt_params_AUA_Q,opt_params_2CLJ)
        #Propose and do an RJMC move (either of parameter or model type, and record the outcome)
        
        if model_swap_attempt=='True':
            if model_swap_type=='01':
                alpha_vector_01.append(alpha)
            elif model_swap_type=='10':
                alpha_vector_10.append(alpha)
            elif model_swap_type=='02':
                alpha_vector_02.append(alpha)            
            elif model_swap_type=='20':
                alpha_vector_20.append(alpha)        
        
        if acceptance == 'True':
            accept_vector[i]=1
        logp_trace[i+1] = new_log_prob
        trace[i+1] = new_params
        percent_deviation_trace[i+1]=computePercentDeviations(compound_2CLJ,thermo_data_rhoL[:,0],thermo_data_Pv[:,0],thermo_data_SurfTens[:,0],trace[i+1],thermo_data_rhoL[:,1],thermo_data_Pv[:,1],thermo_data_SurfTens[:,1],Tc_lit[0],rhol_hat_models,Psat_hat_models,SurfTens_hat_models,T_c_hat_models)
        
        if (not (i+1) % tune_freq) and (i < tune_for):
        #Do parameter move tuning with specified frequency and length
            #print('Tuning on step %1.1i' %i)
            #print(np.sum(accept_vector[i-tune_freq:]))
            acceptance_rate = np.sum(accept_vector)/i           
            #print(acceptance_rate)
            for m in range (n_params-1):
                if acceptance_rate<0.2:
                    prop_sd[m+1] *= 0.9
                    #print('Yes')
                elif acceptance_rate>0.5:
                    prop_sd[m+1] *= 1.1
                    #print('No')         
           
    
    alpha_matrix=np.asarray([np.asarray(alpha_vector_01),np.asarray(alpha_vector_10),np.asarray(alpha_vector_02),np.asarray(alpha_vector_20)])
    
    return trace,logp_trace, percent_deviation_trace, attempt_matrix,acceptance_matrix,prop_sd,accept_vector,alpha_matrix

def RJMC_Moves(current_params,current_model,current_log_prob,n_models,swap_freq,n_params,prop_sd,attempt_matrix,acceptance_matrix,jacobian,transition_function,record_acceptance,opt_params_AUA,opt_params_AUA_Q,opt_params_2CLJ):
    
    params = current_params.copy()# This approach updates previous param values
    #Grab a copy of the current params to work with
    #current_log_prob_copy=copy.deepcopy(current_log_prob)
    
    #Roll a dice to decide what kind of move will be suggested
    mov_ran=np.random.random()
    
    model_swap_attempt='False'
    model_swap_type='null'
    
    #swap_freq = Frequency that jumps between models are proposed.  Probably should not be set higher than 0.2 (model swaps are not accepted very often and doing a high percentage of them leads to poor sampling)   
    if mov_ran <= swap_freq:
        
        #Do model proposal
        params,rjmc_jacobian,proposed_log_prob,proposed_model,w,lamda,transition_function=model_proposal(current_model,n_models,n_params,params,jacobian,transition_function,opt_params_AUA,opt_params_AUA_Q,opt_params_2CLJ)
        
        alpha = (proposed_log_prob - current_log_prob) + np.log(rjmc_jacobian) + np.log(transition_function)
        
        BAR_ratio = (proposed_log_prob - current_log_prob) + np.log(rjmc_jacobian)
        acceptance=accept_reject(alpha)
        #Accept or reject proposal and record new parameters/metadata
        
        model_swap_attempt='True'
        model_swap_type=str(proposed_model)+str(current_model)
        
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
                
    else:
        #Propose parameter swap
        params,proposed_log_prob=parameter_proposal(params,n_params,prop_sd)    
        
        alpha = (proposed_log_prob - current_log_prob)
        BAR_ratio=0
        acceptance=accept_reject(alpha)
        #Accept or reject proposal and record new parameters/metadata
    
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
    
                   
    return new_params,new_log_prob,attempt_matrix,acceptance_matrix,acceptance,-BAR_ratio,model_swap_attempt,model_swap_type 

def accept_reject(alpha):    
    urv=runif()
    #Metropolis-Hastings accept/reject criteria
    if np.log(urv) < alpha:  
        acceptance='True'
    else: 
        acceptance='False'
    return acceptance
        
def model_proposal(current_model,n_models,n_params,params,jacobian,transition_function,opt_params_AUA,opt_params_AUA_Q,opt_params_2CLJ):
    proposed_model=copy.deepcopy(current_model)
    
    #Propose new model to jump to
    while proposed_model==current_model:
        proposed_model=int(np.floor(np.random.random()*n_models))
        if proposed_model==2 and current_model==1:
            proposed_model=copy.deepcopy(current_model)
        elif proposed_model==1 and current_model==2:
            proposed_model=copy.deepcopy(current_model)
    lamda=5
    params[0] = proposed_model
    w=1
    if proposed_model==1 and current_model==0:
        
        #AUA ---> AUA+Q
        
        #Optimum Matching
        #params[1] = (opt_params_AUA_Q[0]/opt_params_AUA[0])*params[1]
        #params[2] = (opt_params_AUA_Q[1]/opt_params_AUA[1])*params[2]
        #params[3] = (opt_params_AUA_Q[2]/opt_params_AUA[2])*params[3]
        
        
        w=runif()
        
        #THIS IS IMPORTANT needs to be different depending on which direction
        
        #params[4]=w*2
        params[4] = -(1/lamda)*np.log(w)
        #Propose a value of Q from an exponential distribution using the inverse CDF method (this is nice because it keeps the transition probability simple)


    elif proposed_model==0 and current_model==1:
        
        #AUA+Q ----> AUA
        
        #Optimum Matching
        #params[1] = (opt_params_AUA[0]/opt_params_AUA_Q[0])*params[1]
        #params[2] = (opt_params_AUA[1]/opt_params_AUA_Q[1])*params[2]
        #params[3] = (opt_params_AUA[2]/opt_params_AUA_Q[2])*params[3]
        
        #w=params[4]/2
        
        #Still need to calculate what "w" (dummy variable) would be even though we don't use it (to satisfy detailed balance)
        w=np.exp(-lamda*params[4])
        
        
        
        
        params[4] = 0
        
    elif proposed_model==2 and current_model==0:
        
        #AUA--->UA
        
        params[1] = (opt_params_2CLJ[0]/opt_params_AUA[0])*params[1]
        params[2] = (opt_params_2CLJ[1]/opt_params_AUA[1])*params[2]
        params[3] = opt_params_2CLJ[2]
        
        params[4] = 0
        w=1
        
        
    elif proposed_model==0 and current_model==2:
        #UA ----> AUA
    
        params[1] = (opt_params_AUA[0]/opt_params_2CLJ[0])*params[1]
        params[2] = (opt_params_AUA[1]/opt_params_2CLJ[1])*params[2]
        params[3] = (opt_params_AUA[2]/opt_params_2CLJ[2])*params[3]    
        w=1
        params[4] = 0

    proposed_log_prob=calc_posterior(*params)
    jacobian =  jacobian(n_models,n_params,w,lamda,opt_params_AUA,opt_params_AUA_Q,opt_params_2CLJ)
    rjmc_jacobian=jacobian[current_model,proposed_model]
    transition_function=transition_function(n_models,w)
    transition_function=transition_function[current_model,proposed_model]
    #Return values of jacobian in order to properly calculate accept/reject
    return params,rjmc_jacobian,proposed_log_prob,proposed_model,w,lamda,transition_function


def parameter_proposal(params,n_params,prop_sd):
    
    #Choose a random parameter to change
    if params[0] == 0:
        proposed_param=int(np.ceil(np.random.random()*(n_params-2)))
    elif params[0] == 1:
        proposed_param=int(np.ceil(np.random.random()*(n_params-1)))
    elif params[0] == 2:
        proposed_param=int(np.ceil(np.random.random()*(n_params-3)))
    
   
        
    params[proposed_param] = rnorm(params[proposed_param], prop_sd[proposed_param])
    proposed_log_prob=calc_posterior(*params)

    return params, proposed_log_prob

guess_params=np.zeros((3,np.size(guess_0)))
guess_params[0,:]=guess_0
guess_params[1,:]=guess_1
guess_params[2,:]=guess_2


initial_sd = [1,2, 0.01,0.01,0.5]
guess_sd=np.zeros((3,np.size(guess_0)))
guess_sd[0,:]=initial_sd
guess_sd[1,:]=initial_sd
guess_sd[2,:]=initial_sd
n_models=3


'''
def mcmc_prior_proposal(n_models,calc_posterior,guess_params,guess_sd):
    swap_freq=0.0
    n_iter=200000
    tune_freq=100
    tune_for=10000
    parameter_prior_proposal=np.empty((n_models,np.size(guess_params,1),2))

    for i in range(1,n_models):
        initial_values=guess_params[i,:]
        initial_sd=guess_sd[i,:]
        trace,logp_trace,percent_deviation_trace, attempt_matrix,acceptance_matrix,prop_sd,accept_vector = RJMC_outerloop(calc_posterior,n_iter,initial_values,initial_sd,n_models,swap_freq,tune_freq,tune_for,1,1,1,1,1)
        trace_tuned = trace[tune_for:]
        max_ap=np.zeros(np.size(trace_tuned,1))
        map_CI=np.zeros((np.size(trace_tuned,1),2))

        for j in range(np.size(trace_tuned,1)):
            bins,values=np.histogram(trace_tuned[:,j],bins=100)
            max_ap[j]=(values[np.argmax(bins)+1]+values[np.argmax(bins)])/2
            map_CI[j]=hpd(trace_tuned[:,j],alpha=0.05)
            sigma_hat=np.sqrt(map_CI[j,1]-map_CI[j,0])/(2*1.96)
            parameter_prior_proposal[i,j]=[max_ap[j],sigma_hat]
    return parameter_prior_proposal,trace_tuned



#parameter_prior_proposals,trace_tuned=mcmc_prior_proposal(n_models,calc_posterior,guess_params,guess_sd)
'''
n_models=3
#Number of models considered

initial_type=sys.argv[10]

if initial_type =='AUA':
    initial_values=guess_0
elif initial_type == 'AUA+Q':
    initial_values=guess_1
elif initial_type == 'UA':
    initial_values = guess_2
else:
    #initial_values=np.asarray([random.randint(0,n_models-1),logitrvs(*eps_prior),logitrvs(*sig_prior),logitrvs(*L_prior),gammarvs(*Q_prior)])
    initial_values=np.asarray([random.randint(0,n_models-1),uniformrvs(*eps_prior),uniformrvs(*sig_prior),uniformrvs(*L_prior),gammarvs(*Q_prior)])
    if initial_values[0]==0:
        initial_values[4]=0
    elif initial_values[0]==2:
        initial_values[3]=NIST_bondlength
        initial_values[4]=0
print(initial_values)

guess_test=[1,91.4,0.363,0.148,0]


initial_values=np.empty(5)
#randomly generate initial values from prior distributions

#initial_values=np.asarray([random.randint(0,n_models-1),logitrvs(*eps_prior),logitrvs(*sig_prior),logitrvs(*L_prior),gammarvs(*Q_prior)])
initial_values=np.asarray([random.randint(0,n_models-1),uniformrvs(*eps_prior),uniformrvs(*sig_prior),uniformrvs(*L_prior),gammarvs(*Q_prior)])



    


#guess_1 # Can use critical constants
initial_sd = np.asarray(initial_values)/100


n_iter=int(sys.argv[7])
#Number of iterations.  Should get decent results at 10^6, better at 10^7 (but takes like 3-5 hours)

tune_freq=100
tune_for=10000
#Tuning params



swap_freq=float(sys.argv[6])
#Frequency of proposed model swaps. Best to keep below 0.2
#Definitely a tradeoff between too low (slow convergence of sampling ratio) and too high (poor sampling in general).  Have found 0.05-0.1 to be good


print('Compound: '+compound)
print('Properties: '+properties)
print('MCMC Steps: '+str(n_iter))


trace,logp_trace,percent_deviation_trace, attempt_matrix,acceptance_matrix,prop_sd,accept_vector,alpha_vector = RJMC_outerloop(calc_posterior,n_iter,initial_values,initial_sd,n_models,swap_freq,tune_freq,tune_for,jacobian,transition_function,opt_params_AUA,opt_params_AUA_Q,opt_params_2CLJ)
#Initiate sampling!
label=sys.argv[9]
directory=sys.argv[8]
print(directory)

#%%
# POST PROCESSING

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

fname=compound+'likelihood_data_amount_10'+'_'+properties+'_'+str(n_points)+'_'+str(n_iter)+'_'+str(date.today())

lit_params,lit_devs=import_literature_values(number_criteria,compound)
#new_lit_devs=computePercentDeviations(thermo_data_rhoL[:,0],thermo_data_Pv[:,0],thermo_data_SurfTens[:,0],lit_devs,thermo_data_rhoL[:,1],thermo_data_Pv[:,1],thermo_data_SurfTens[:,1],Tc_lit[0],rhol_hat_models,Psat_hat_models,SurfTens_hat_models,T_c_hat_models)

#%%
new_lit_devs=recompute_lit_percent_devs(lit_params,computePercentDeviations,thermo_data_rhoL[:,0],thermo_data_Pv[:,0],thermo_data_SurfTens[:,0],lit_devs,thermo_data_rhoL[:,1],thermo_data_Pv[:,1],thermo_data_SurfTens[:,1],Tc_lit[0],rhol_hat_models,Psat_hat_models,SurfTens_hat_models,T_c_hat_models,compound_2CLJ)
pareto_point,pareto_point_values=findParetoPoints(percent_deviation_trace_tuned,trace_tuned,0)


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

Exp_ratio=prob_0/prob_1

'''
BAR_estimate=BAR(np.asarray(alpha_vector[0]),alpha_vector[1])

BF_BAR=np.exp(-BAR_estimate[0])
BF_BAR_LB=np.exp(-(BAR_estimate[0]+BAR_estimate[1]))
BF_BAR_UB=np.exp(-(BAR_estimate[0]-BAR_estimate[1]))
print(BF_BAR)
print(BF_BAR_LB,BF_BAR_UB)
'''
#plot_bar_chart(prob,fname,properties,compound,n_iter,n_models)

#create_percent_dev_triangle_plot(percent_deviation_trace_tuned,fname,'percent_dev_trace',new_lit_devs,prob,properties,compound,n_iter)

#print('Analytical sampling ratio: %2.3f' % ratio)
print('Experimental sampling ratio: %2.3f' % Exp_ratio )


print('Detailed Balance')

#These sets of numbers should be roughly equal to each other (If both models are sampled).  If not, big problem 

print(prob_0*transition_matrix[0,1])
print(prob_1*transition_matrix[1,0])
    
print(prob_0*transition_matrix[0,2])
print(prob_2*transition_matrix[2,0])

print(prob_1*transition_matrix[1,2])
print(prob_2*transition_matrix[2,1])

#trace_tuned=np.load('trace/trace_C2H6_All_10_50000000_2019-03-08.npy')

trace_model_0=[]
trace_model_1=[]
trace_model_2=[]
log_trace_0=[]
log_trace_1=[]
log_trace_2=[]

#Initiate data frames for separating model traces

plt.plot(logp_trace,label='Log Posterior')
plt.legend()
plt.show()


plt.plot(trace[:,0])
plt.show()

#np.save('trace/trace_'+fname+'.npy',trace_tuned)
#np.save('logprob/logprob_'+fname+'.npy',logp_trace)
#np.save('percent_dev/percent_dev_'+fname+'.npy',percent_deviation_trace_tuned)
#Save trajectories (can be disabled since they are big files)


for i in range(np.size(trace_tuned,0)):
    if trace_tuned[i,0] == 0:
        trace_model_0.append(trace_tuned[i])
        #log_trace_0.append(logp_trace[i])
    elif trace_tuned[i,0] == 1:
        trace_model_1.append(trace_tuned[i])
        #log_trace_1.append(logp_trace[i])
    elif trace_tuned[i,0] == 2:
        trace_model_2.append(trace_tuned[i])
        #log_trace_2.append(logp_trace[i])        
        
        
trace_model_0=np.asarray(trace_model_0)
trace_model_1=np.asarray(trace_model_1)
trace_model_2=np.asarray(trace_model_2)

#plt.hist(alpha_vector[0],range=[-10,100],bins=50,alpha=0.7)
#plt.hist(alpha_vector[1],range=[-10,100],bins=50,alpha=0.7)
#plt.show()
sys.exit(np.mean(logp_trace_tuned))
#create_param_triangle_plot_4D(trace_model_0,fname,'trace_model_0',lit_params,properties,compound,n_iter,sig_prior,eps_prior,L_prior,Q_prior)
#create_param_triangle_plot_4D(trace_model_1,fname,'trace_model_1',lit_params,properties,compound,n_iter,sig_prior,eps_prior,L_prior,Q_prior)
#create_param_triangle_plot_4D(trace_model_2,fname,'trace_model_2',lit_params,properties,compound,n_iter,sig_prior,eps_prior,L_prior,Q_prior)

#Plot parameters

get_metadata(directory,label,compound,properties,sig_prior,eps_prior,L_prior,Q_prior,n_iter,swap_freq,n_points,transition_matrix,prob,attempt_matrix,acceptance_matrix)

#write outputs to file
