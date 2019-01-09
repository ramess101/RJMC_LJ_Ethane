#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 11:33:46 2018

@author: owenmadin
"""

"""
This code performs RJMC with Bayesian inference between two square regions of (x,y) probability space with the same probability density.
The regions are defined as the square with corners (0,0),(0,1),(1,0),(1,1) (Region 1) and the square with corners (2,2),(2,3),(3,2),(3,3) (Region 2).
The probability density function is uniformly defined as f(x,y)=1 on both models, for now.  With a naive RJMC model
(no mapping between regions), cross model jumps should not be possible.  We will explore two types of mappings: a translational mapping 
(adding (2,2) to every point in region 1 and subtracting it from every point in region 2.  The second type of mapping is a scaling mapping,
which maps the center of the first region to the center of the second region by multiplying each coordinate by 3, allowing for visitation to the whole
space.  We expect both regions to be sampled equally (prior of both models equally likely) but that the scaling map will have a much lower acceptance ratio.
    
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from LennardJones_correlations import LennardJones
from LennardJones_2Center_correlations import LennardJones_2C
from scipy.stats import distributions
from scipy.stats import linregress
from scipy.stats import multivariate_normal as mvn
from scipy.optimize import minimize
import random as rm

# Define probabilities for both regions


cov_matrix_0=[[1,0],[0,1]]
cov_matrix_1=[[1,0],[0,1]]

def test_pdf(model,x,y,cov_matrix_0,cov_matrix_1):
    if model == 0:
        rv=mvn([10,2],cov_matrix_0)
        f=rv.pdf([x,y])
    if model == 1:
        rv=mvn([2,10],cov_matrix_1)
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
    logp += duni(x, -20, 40)
    logp += duni(y, -20, 40) 
    prop_density=test_pdf(model,x,y,cov_matrix_0,cov_matrix_1)
    logp += np.log(prop_density)
    
    return logp

def T_matrix_scale():
    T_matrix_x_scale=np.ones((2,2))
    T_matrix_y_scale=np.ones((2,2))
    T_matrix_x_scale[0,1]=2/10
    T_matrix_y_scale[0,1]=10/2
    T_matrix_x_scale[1,0]=10/2
    T_matrix_y_scale[1,0]=2/10
    return T_matrix_x_scale, T_matrix_y_scale


def T_matrix_translation():
    T_matrix_x=np.zeros((2,2))
    T_matrix_y=np.zeros((2,2))
    T_matrix_x[0,1]=-5
    T_matrix_y[0,1]=-5
    T_matrix_x[1,0]=5
    T_matrix_y[1,0]=5
    return T_matrix_x, T_matrix_y
    
T_matrix_x, T_matrix_y = T_matrix_translation()
T_matrix_x_scale, T_matrix_y_scale = T_matrix_scale()

'''
def accept_reject(alpha,params):
    
    urv = runif()
    
    if np.log(urv) < alpha:
    
    # Accept
    current_params = params
    logp_trace_current = proposed_log_prob.copy()
    current_log_prob = proposed_log_prob.copy()

def RJMC__outer_loop
'''

def RJMC_tuned(calc_posterior,n_iterations, initial_values, prop_var, 
                     tune_for=None, tune_interval=1, map_scale='False'):
    
    n_params = len(initial_values) #One column is the model number
            
    # Initial proposal standard deviations
    prop_sd = prop_var
    
    #Initialize matrices to count number of moves of each type
    attempt_matrix=np.zeros((n_params-1,n_params-1))
    acceptance_matrix=np.zeros((n_params-1,n_params-1))

    
    # Initialize trace for parameters
    trace = np.zeros((n_iterations+1, n_params)) #n_iterations + 1 to account for guess
    logp_trace = np.zeros(n_iterations+1)
    # Set initial values
    trace[0] = initial_values

    # Initialize acceptance counts
    accepted = [0]*n_params
    rejected = [0]*n_params
               
    model_swaps = 0
    model_swap_attempts = 0
    swap_freq = 10
    swap_flag='False'
    # OCM: Currently attempting a model swap every single move, although this can be easily changed.  This is something that is not of critical importance now but will be important in the future.
    
    # Calculate joint posterior for initial values
    current_log_prob = calc_posterior(*trace[0])
    
    logp_trace[0] = current_log_prob
    #OCM: This is just the priors at this point.
    
    if tune_for is None:
        tune_for = n_iterations/2
    
    for i in range(n_iterations):
        swap_flag='False'
        if not i%1000: print('Iteration '+str(i))
    
        # Grab current parameter values
        current_params = trace[i].copy()
        trace[i+1] = current_params.copy() #Initialize the next step with the current step. Then update if MCMC move is accepted
        current_model = int(current_params[0])
        logp_trace[i+1] = current_log_prob.copy()
        
        # Loop through model parameters
        
        for j in range(n_params):
            
            # Get current value for parameter j
            params = current_params.copy() # This approach updates previous param values
            
            # Propose new values
            if j == 0: #If proposing a new model
                if not i%swap_freq:
                    mod_ran = np.random.random()
                    if mod_ran < 1./2: #Use new models with equal probability
                        proposed_model = 0
                    elif mod_ran >= 1./2:
                        proposed_model = 1
                    if proposed_model != current_model:
                        model_swap_attempts += 1
                        params[0] = proposed_model
                        if map_scale=='True':
                            params[1] *= T_matrix_x_scale[current_model,proposed_model]
                            params[2] *= T_matrix_y_scale[current_model,proposed_model]
                        else:   
                            params[1] += T_matrix_x[current_model,proposed_model]
                            params[2] += T_matrix_y[current_model,proposed_model]
                        # Calculate log posterior with proposed value
                        proposed_log_prob = calc_posterior(*params)
                        attempt_matrix[current_model,proposed_model]+=1
                        # Log-acceptance rate
                        alpha = (proposed_log_prob - current_log_prob) + np.log(T_matrix_x_scale[current_model,proposed_model]) + np.log(T_matrix_y_scale[current_model,proposed_model])
                        
                        urv = runif()
    
                        # Test proposed value
                        if np.log(urv) < alpha:
                            
                            # Accept
                            trace[i+1] = params
                            logp_trace[i+1] = proposed_log_prob.copy()
                            current_log_prob = proposed_log_prob.copy()
                            current_params = params
                            
                            
                            #accepted[j] += 1
                            if j == 0:
                                if proposed_model != current_model:
                                    model_swaps += 1
                                    swap_flag = 'True'
            else:
                if swap_flag=='False':
                    params[j] = rnorm(current_params[j], prop_sd[j])
            
            attempt_matrix[current_model,current_model]+=1
        
            # Calculate log posterior with proposed value
            proposed_log_prob = calc_posterior(*params)
    
            # Log-acceptance rate
            alpha = (proposed_log_prob - current_log_prob)
    

        #OCM:  The two components of the acceptance ratio here are the log of the ratio of the probabilities, and the log of the jacobian determinant between the model spaces
    
    
    
    
     
            # Sample a uniform random variate (urv)
            urv = runif()
    
            # Test proposed value
            if swap_flag == 'False':
                if np.log(urv) < alpha:
                
                    # Accept
                    trace[i+1] = params
                    logp_trace[i+1] = proposed_log_prob.copy()
                    current_log_prob = proposed_log_prob.copy()
                    current_params = params
                    accepted[j] += 1
                    acceptance_matrix[current_model,current_model]+=1
                    #if j == 0:
                    #   if proposed_model != current_model:
                    #      model_swaps += 1
                        
                else:
                    # Reject
                    rejected[j] += 1
    
    '''
            # Tune every 100 iterations
            if (not (i+1) % tune_interval) and (i < tune_for) and j != 0:

                acceptance_rate = (1.*accepted[j])/tune_interval             
                if acceptance_rate<0.2:
                    prop_sd[j] *= 0.9
                elif acceptance_rate>0.5:
                    prop_sd[j] *= 1.1                  

                #print(prop_sd[j])
                accepted[j] = 0              
'''
    accept_prod = np.array(accepted)/(np.array(accepted)+np.array(rejected))                    

    print('Proposed standard deviations are: '+str(prop_sd))
                
    return trace, trace[tune_for:], logp_trace, logp_trace[tune_for:],accept_prod, model_swaps, model_swap_attempts,acceptance_matrix,attempt_matrix




# Set the number of iterations to run RJMC and how long to tune for
n_iter = 50000 # 20000 appears to be sufficient
tune_for = 10000 #10000 appears to be sufficient
guess_0=[0,5,5]
guess_var=[1,1,1]
trace_all,trace_tuned,logp_all,logp_tuned, acc_tuned, model_swaps, model_swap_attempts,acceptance_matrix,attempt_matrix = RJMC_tuned(calc_posterior, n_iter, guess_0, prop_var=guess_var, tune_for=tune_for,map_scale='True')
#%%


'''
print('Acceptance Rate during production for eps, sig: '+str(acc_tuned[1:]))

print('Acceptance model swap during production: '+str(model_swaps/model_swap_attempts))
'''

print('Acceptance Rates')
print(acceptance_matrix/attempt_matrix)
#OCM: Something is wrong with this as it is greater than one, which shouldn't be possible.  Probably just a calculation error that doesn't affect RJMC

model_params = trace_all[:,0]

# Converts the array with number of model parameters into an array with the number of times there was 1 parameter or 2 parameters
model_count = np.array([len(model_params[model_params==0]),len(model_params[model_params==1])])


prob_0 = 1.*model_count[0]/(n_iter)
print('Percent that  model 0 is sampled: '+str(prob_0 * 100.)) #The percent that use 1 parameter model

prob_1 = 1.*model_count[1]/(n_iter)
print('Percent that model 1 is sampled: '+str(prob_1 * 100.)) #The percent that use two center UA LJ
    
trace_model_0=[]
trace_model_1=[]
for i in range(np.size(trace_all,0)):
    if trace_all[i,0] == 0:
        trace_model_0.append(trace_all[i])
    elif trace_all[i,0] == 1:
        trace_model_1.append(trace_all[i])
        
        
trace_model_0=np.asarray(trace_model_0)
trace_model_1=np.asarray(trace_model_1)
f = plt.figure()
plt.scatter(trace_model_0[::10,2],trace_model_0[::10,1],s=1,label='Model 0',marker=',')
plt.scatter(trace_model_1[::10,2],trace_model_1[::10,1],s=1,label='Model 1',marker=',')
plt.title('RJMC between 2-D normals')
plt.legend()
plt.show()
plt.plot(trace_all[:,0],label='Model Choice')
plt.legend()
plt.show()

plt.plot(logp_all,label='Log Posterior')
plt.show()
