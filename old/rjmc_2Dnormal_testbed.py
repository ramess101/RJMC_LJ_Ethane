#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 12:53:35 2018

This is a testbed for reversible monte carlo moves between two different models, as well as normal MCMC intra-model moves
The test models are 2-D normal distributions, which are 

@author: owenmadin
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.stats import distributions
from scipy.stats import linregress
from scipy.stats import multivariate_normal as mvn
from scipy.optimize import minimize
import random as rm
import copy
from pymc3.stats import hpd

# Define probabilities for both regions

data_vec=[5,1,4,1,3,3,1,6,1,1,2,4,3,1,2,1,0,2,2,2,3,2,3,3,1,2,2,3,2,4,1,1,3,3,2,3,3,3,7,7,1,3,3,3,4,3,1,1,7,3,2,2,2,5,1,2,2,3,2,4,1,3,2,6]

cov_matrix_0=[[1,0],[0,1]]
cov_matrix_1=[[1,0],[0,1]]

def test_pdf(model,x,y,cov_matrix_0,cov_matrix_1):
    if model == 0:
        rv=mvn([5,5],cov_matrix_0)
        f=np.log(5)+rv.logpdf([x,y])
    if model == 1:
        rv=mvn([40,40],cov_matrix_1)
        f=rv.logpdf([x,y])
#    if model == 2:
#        rv=mvn([20,20],cov_matrix_1)
#        f=rv.logpdf([x,y])
    return f


normalization_0=(np.sqrt((2*np.pi**2*np.linalg.det(cov_matrix_0))))
normalization_1=(np.sqrt((2*np.pi**2*np.linalg.det(cov_matrix_1))))
ratio=normalization_0/normalization_1

dnorm = distributions.norm.logpdf
dgamma = distributions.gamma.logpdf
duni = distributions.uniform.logpdf

rnorm = np.random.normal
runif = np.random.rand

#Define log priors for the distributions.  For now, we will use a uniform prior on (10,10), but we may want to change the prior for different models in the future

def calc_posterior(model,x,y):
    
    logp = 0
    logp += duni(x, -50, 100)
    logp += duni(y, -50, 100) 
    prop_density=test_pdf(model,x,y,cov_matrix_0,cov_matrix_1)
    logp += prop_density
    
    return logp


def T_matrix_scale():
    T_matrix_x_scale=np.ones((2,2))
    T_matrix_y_scale=np.ones((2,2))
    T_matrix_x_scale[0,1]=40/5
    T_matrix_y_scale[0,1]=40/5
    T_matrix_x_scale[1,0]=5/40
    T_matrix_y_scale[1,0]=5/40
    return T_matrix_x_scale, T_matrix_y_scale

'''
def T_matrix_translation():
    T_matrix_x=np.zeros((2,2))
    T_matrix_y=np.zeros((2,2))
    T_matrix_x[0,1]=2
    T_matrix_y[0,1]=2
    T_matrix_x[1,0]=-2
    T_matrix_y[1,0]=-2
    return T_matrix_x, T_matrix_y
   ''' 
#Not using translations right now, but could as a basic test
#T_matrix_x, T_matrix_y = T_matrix_translation()
T_matrix_x_scale, T_matrix_y_scale = T_matrix_scale()


swap_freq=0.5
#The fraction of times a model swap is suggested as the move, rather than an intra-model move

def RJMC_outerloop(calc_posterior,n_iterations,initial_values,initial_sd,n_models,swap_freq,tune_freq,tune_for,T_matrix_1,T_matrix_2):
    
    
    #INITIAL SETUP FOR MC LOOP
    #-----------------------------------------------------------------------------------------#
    
    n_params = len(initial_values) #One column is the model number
    accept_vector=np.zeros((n_iterations))
    prop_sd=initial_sd
    
    #Initialize matrices to count number of moves of each type
    attempt_matrix=np.zeros((n_params-1,n_params-1))
    acceptance_matrix=np.zeros((n_params-1,n_params-1))
    
    
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
        
        new_params, new_log_prob, attempt_matrix,acceptance_matrix,acceptance = RJMC_Moves(current_params,current_model,current_log_prob,n_models,swap_freq,n_params,prop_sd,attempt_matrix,acceptance_matrix,T_matrix_1,T_matrix_2,record_acceptance)
        
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
            

            
    
    
def RJMC_Moves(current_params,current_model,current_log_prob,n_models,swap_freq,n_params,prop_sd,attempt_matrix,acceptance_matrix,T_matrix_1,T_matrix_2,record_acceptance):
    
    params = current_params.copy()# This approach updates previous param values
    #Grab a copy of the current params to work with
    #current_log_prob_copy=copy.deepcopy(current_log_prob)
    
    #Roll a dice to decide what kind of move will be suggested
    mov_ran=np.random.random()
    
    if mov_ran <= swap_freq:
        
        params,rjmc_jacobian,proposed_log_prob,proposed_model=model_proposal(current_model,n_models,params,T_matrix_1,T_matrix_2)
        
        alpha = (proposed_log_prob - current_log_prob) + rjmc_jacobian
        
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
        
def model_proposal(current_model,n_models,params,T_matrix_1,T_matrix_2):
    proposed_model=current_model
    while proposed_model==current_model:
        proposed_model=int(np.floor(np.random.random()*n_models))
            
    params[0] = proposed_model
    params[1] *= T_matrix_1[current_model,proposed_model]
    params[2] *= T_matrix_2[current_model,proposed_model]

    proposed_log_prob=calc_posterior(*params)
    rjmc_jacobian =  np.log(T_matrix_1[current_model,proposed_model]) + np.log(T_matrix_2[current_model,proposed_model])
    return params,rjmc_jacobian,proposed_log_prob, proposed_model
    #Switch models and map parameters to new distributions
    
    
    
def parameter_proposal(params,n_params,prop_sd):
    proposed_param=int(np.ceil(np.random.random()*(n_params-1)))
    params[proposed_param] = rnorm(params[proposed_param], prop_sd[proposed_param])
    proposed_log_prob=calc_posterior(*params)
    return params, proposed_log_prob
        

initial_values=[1,15,15]
initial_sd=[1,1,1]
n_iterations=50000
tune_freq=100
tune_for=10000
n_models=2

trace,logp_trace,attempt_matrix,acceptance_matrix,prop_sd,accept_vector = RJMC_outerloop(calc_posterior,n_iterations,initial_values,initial_sd,n_models,swap_freq,tune_freq,tune_for,T_matrix_x_scale,T_matrix_y_scale)

#%%
# POST PROCESSING

print('Attempted Moves')
print(attempt_matrix)

prob_matrix=acceptance_matrix/attempt_matrix
transition_matrix=np.ones((2,2))
transition_matrix[0,0]=1-prob_matrix[0,1]
transition_matrix[0,1]=prob_matrix[0,1]
transition_matrix[1,0]=prob_matrix[1,0]
transition_matrix[1,1]=1-prob_matrix[1,0]
print('Transition Matrix:')
print(transition_matrix)
trace_tuned = trace[tune_for:]
model_params = trace_tuned[:,0]

# Converts the array with number of model parameters into an array with the number of times there was 1 parameter or 2 parameters
model_count = np.array([len(model_params[model_params==0]),len(model_params[model_params==1])])


prob_0 = 1.*model_count[0]/(n_iterations-tune_for)
print('Percent that  model 0 is sampled: '+str(prob_0 * 100.)) #The percent that use 1 parameter model

prob_1 = 1.*model_count[1]/(n_iterations-tune_for)
print('Percent that model 1 is sampled: '+str(prob_1 * 100.)) #The percent that use two center UA LJ

Exp_ratio=prob_0/prob_1

print('Analytical sampling ratio: %2.3f' % ratio)
print('Experimental sampling ratio: %2.3f' % Exp_ratio )


print('Detailed Balance')
print(prob_0*prob_matrix[0,1])
print(prob_1*prob_matrix[1,0])
    
trace_model_0=[]
trace_model_1=[]
for i in range(np.size(trace_tuned,0)):
    if trace_tuned[i,0] == 0:
        trace_model_0.append(trace_tuned[i])
    elif trace_tuned[i,0] == 1:
        trace_model_1.append(trace_tuned[i])
        
        
trace_model_0=np.asarray(trace_model_0)
trace_model_1=np.asarray(trace_model_1)
f = plt.figure()
plt.scatter(trace_model_0[::10,1],trace_model_0[::10,2],s=1,label='Model 0',marker=',',alpha=0.7)
plt.scatter(trace_model_1[::10,1],trace_model_1[::10,2],s=1,label='Model 1',marker=',',alpha=0.7)
plt.legend()
plt.show()

plt.plot(trace[::500,1],trace[::500,2])
plt.show()
map_x_0=hpd(trace_model_0[:,1],alpha=0.05)
map_x_1=hpd(trace_model_1[:,1],alpha=0.05)
map_y_0=hpd(trace_model_0[:,2],alpha=0.05)
map_y_1=hpd(trace_model_1[:,2],alpha=0.05)
CI_x_0=map_x_0[1]-map_x_0[0]
CI_x_1=map_x_1[1]-map_x_1[0]
CI_y_0=map_y_0[1]-map_y_0[0]
CI_y_1=map_y_1[1]-map_y_1[0]

#trace_model_0_subsample=trace_model_0[::1000]
#trace_model_1_subsample=trace_model_1[::1000]
#trace_subsample=trace_tuned[::1000]
#Try subsampling to make the graphs look better.
plt.hist(trace_model_0[:,1],bins=100,label='x values Model 0',density=True)


plt.hist(trace_model_1[:,1],bins=100,label='x values Model 1',density=True)

plt.legend()
plt.show()


plt.hist(trace_model_0[:,2],bins=100,label='y values Model 0',density=True)
plt.hist(trace_model_1[:,2],bins=100,label='y values Model 1',density=True)
plt.legend()
plt.show()


plt.plot(trace_tuned[:,0],label='Model Choice')
plt.legend()
plt.show()

plt.plot(logp_trace,label='Log Posterior')
plt.legend()
plt.show()

swap01=0
swap10=0
same=0

for i in range(np.size(logp_trace)-1):

    if trace[i+1][0] < trace[i][0]:
        swap10+=1
    elif trace[i+1][0] > trace[i][0]:
        swap01+=1
    else:
        same+=1