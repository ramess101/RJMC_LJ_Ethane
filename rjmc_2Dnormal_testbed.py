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

# Define probabilities for both regions

def test_pdf(model,x,y):
    if model == 0:
        rv=mvn([2,10],[[0.1,0],[0,0.1]])
        f=rv.pdf([x,y])
    if model == 1:
        rv=mvn([10,2],[[0.1,0],[0,0.1]])
        f=rv.pdf([x,y])   
    return f

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
    prop_density=test_pdf(model,x,y)
    logp += np.log(prop_density)
    
    return logp


def T_matrix_scale():
    T_matrix_x_scale=np.ones((2,2))
    T_matrix_y_scale=np.ones((2,2))
    T_matrix_x_scale[0,1]=10/2
    T_matrix_y_scale[0,1]=2/10
    T_matrix_x_scale[1,0]=2/10
    T_matrix_y_scale[1,0]=10/2
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

def RJMC_outerloop(calc_posterior,n_iterations,initial_values,initial_sd,n_models,swap_freq):
    
    
    #INITIAL SETUP FOR MC LOOP
    #-----------------------------------------------------------------------------------------#
    
    n_params = len(initial_values) #One column is the model number
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
    #----------------------------------------------------------------------------------------#
    
    #OUTER MCMC LOOP
    
    for i in range(n_iterations):
        if not i%5000: print('Iteration '+str(i))
        
        
        # Grab current parameter values
        current_params = trace[i].copy()
        trace[i+1] = current_params.copy() #Initialize the next step with the current step. Then update if MCMC move is accepted
        current_model = int(current_params[0])
        logp_trace[i+1] = current_log_prob.copy()
        
        
        
        new_params, new_log_prob, acceptance,attempt_matrix,acceptance_matrix = RJMC_Moves(current_params,current_model,current_log_prob,n_models,swap_freq,n_params,prop_sd,attempt_matrix,acceptance_matrix)
        
        
        if acceptance == 'True':
            logp_trace[i+1] = new_log_prob
            trace[i+1] = new_params
    return trace,logp_trace, attempt_matrix,acceptance_matrix
            
        
    
    
def RJMC_Moves(current_params,current_model,current_log_prob,n_models,swap_freq,n_params,prop_sd,attempt_matrix,acceptance_matrix):
    
    params = current_params.copy() # This approach updates previous param values
    #Grab a copy of the current params to work with
    
    #Roll a dice to decide what kind of move will be suggested
    mov_ran=np.random.random()
    
    if mov_ran <= swap_freq:
        move_type = 'Swap'
    else: 
        move_type = 'Trad'
    
        
    if move_type == 'Swap':
        proposed_model=int(np.floor(np.random.random()*n_models))
        if proposed_model == current_model:
            move_type = 'Trad'
            #Fix this later, should grab another proposed model
        else:
            #Propose new values for the model and map the current values to the new model
            params[0] = proposed_model
            params[1] *= T_matrix_x_scale[current_model,proposed_model]
            params[2] *= T_matrix_y_scale[current_model,proposed_model]
            
            #Accept or Reject this proposed model with the metropolis-hastings acceptance criteria
            new_params,new_log_prob,acceptance,attempt_matrix,acceptance_matrix = accept_reject(current_log_prob,params,current_model,proposed_model,T_matrix_x_scale,T_matrix_y_scale,attempt_matrix,acceptance_matrix)
            
            
            
    if move_type == 'Trad':
        for j in range(n_params-1):
            params[j+1] = rnorm(params[j+1], prop_sd[j+1])
            
        new_params,new_log_prob,acceptance,attempt_matrix,acceptance_matrix = accept_reject(current_log_prob,params,current_model,current_model,T_matrix_x_scale,T_matrix_y_scale,attempt_matrix,acceptance_matrix)
            
        
            #Don't update the first parameter (model number) 
            #Find a better way of doing this in the future
                   
    return new_params,new_log_prob,acceptance,attempt_matrix,acceptance_matrix
            
            
            
def accept_reject(current_log_prob,params,current_model,proposed_model,T_matrix_x,T_matrix_y,attempt_matrix,acceptance_matrix):
    
    attempt_matrix[current_model,proposed_model]+=1
    
    
    proposed_log_prob = calc_posterior(*params)
    
    log_jacobian = np.log(T_matrix_x[current_model,proposed_model]) + np.log(T_matrix_y[current_model,proposed_model])
    
    alpha = (proposed_log_prob - current_log_prob) + log_jacobian
    
    
    urv = runif()
    
    if np.log(urv) < alpha:
    
        # Accept
        new_params = params
        new_log_prob = proposed_log_prob.copy()

        acceptance='True'
        acceptance_matrix[current_model,proposed_model]+=1
    else: 
        #This should be different, doesn't matter for now
        new_params = params
        new_log_prob = proposed_log_prob.copy()
        acceptance='False'

    
    return new_params,new_log_prob,acceptance,attempt_matrix,acceptance_matrix


initial_values=[0,0,0]
initial_sd=[1,0.5,0.5]
n_iterations=100000
n_models=2

trace,logp_trace,attempt_matrix,acceptance_matrix = RJMC_outerloop(calc_posterior,n_iterations,initial_values,initial_sd,n_models,swap_freq)

#%%
# POST PROCESSING

print('Attempted Moves')
print(attempt_matrix)

prob_matrix=acceptance_matrix/attempt_matrix
print('Acceptance Rates:')
print(prob_matrix)

model_params = trace[:,0]

# Converts the array with number of model parameters into an array with the number of times there was 1 parameter or 2 parameters
model_count = np.array([len(model_params[model_params==0]),len(model_params[model_params==1])])


prob_0 = 1.*model_count[0]/(n_iterations)
print('Percent that  model 0 is sampled: '+str(prob_0 * 100.)) #The percent that use 1 parameter model

prob_1 = 1.*model_count[1]/(n_iterations)
print('Percent that model 1 is sampled: '+str(prob_1 * 100.)) #The percent that use two center UA LJ


print('Detailed Balance')
print(prob_0*prob_matrix[0,1])
print(prob_1*prob_matrix[1,0])
    
trace_model_0=[]
trace_model_1=[]
for i in range(np.size(trace,0)):
    if trace[i,0] == 0:
        trace_model_0.append(trace[i])
    elif trace[i,0] == 1:
        trace_model_1.append(trace[i])
        
        
trace_model_0=np.asarray(trace_model_0)
trace_model_1=np.asarray(trace_model_1)
f = plt.figure()
plt.scatter(trace_model_0[:,2],trace_model_0[:,1],s=1,label='Model 0',marker=',',alpha=0.7)
plt.scatter(trace_model_1[:,2],trace_model_1[:,1],s=1,label='Model 1',marker=',',alpha=0.7)
plt.legend()
plt.show()

plt.hist(trace[:,1],bins=50,label='x values')
plt.legend()
plt.show()


plt.hist(trace[:,2],bins=50,label='y values')
plt.legend()
plt.show()


plt.plot(trace[:,0],label='Model Choice')
plt.legend()
plt.show()

plt.plot(logp_trace,label='Log Posterior')
plt.legend()
plt.show()


