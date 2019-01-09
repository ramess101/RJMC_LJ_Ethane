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
    
        Psat_hat = Ethane_2CLJ.Psat_hat_2CLJQ(Temp,eps,sig,L,Q) 
        
    elif model == 1: #Two center AUA LJ+Q
    
        Psat_hat = Ethane_2CLJ.Psat_hat_2CLJQ(Temp,eps,sig,L,Q) 
        
    return Psat_hat #[kPa]       


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

T_min = 137
T_max = 260

rhol_data = rhol_data[T_rhol_data>T_min]
T_rhol_data = T_rhol_data[T_rhol_data>T_min]
rhol_data = rhol_data[T_rhol_data<T_max]
T_rhol_data = T_rhol_data[T_rhol_data<T_max]

Psat_data = Psat_data[T_Psat_data>T_min]
T_Psat_data = T_Psat_data[T_Psat_data>T_min]
Psat_data = Psat_data[T_Psat_data<T_max]
T_Psat_data = T_Psat_data[T_Psat_data<T_max]

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
u_rhol = rhol_data*pu_rhol/100.
u_Psat = Psat_data*pu_Psat/100.
   
# Calculate the estimated standard deviation
sd_rhol = u_rhol/2.
sd_Psat = u_Psat/2.

# Calculate the precision in each property
t_rhol = np.sqrt(1./sd_rhol)
t_Psat = np.sqrt(1./sd_Psat)                                   

# Initial values for the Markov Chain

guess_0 = [0,eps_lit_AUA,sig_lit_AUA,Lbond_lit_AUA,0]
guess_1 = [1,eps_lit2_AUA,sig_lit2_AUA,Lbond_lit2_AUA,Q_lit2_AUA]

## These transition matrices are designed for when rhol is the only target property
#
#Tmatrix_eps = np.ones([3,3])
#Tmatrix_eps[0,1] = eps_lit_UA/eps_lit_LJ
#Tmatrix_eps[0,2] = eps_lit_AUA/eps_lit_LJ
#Tmatrix_eps[1,0] = eps_lit_LJ/eps_lit_UA
#Tmatrix_eps[1,2] = eps_lit_AUA/eps_lit_UA
#Tmatrix_eps[2,0] = eps_lit_LJ/eps_lit_AUA
#Tmatrix_eps[2,1] = eps_lit_UA/eps_lit_AUA
#           
#Tmatrix_sig = np.ones([3,3])
#Tmatrix_sig[0,1] = sig_lit_UA/sig_lit_LJ
#Tmatrix_sig[0,2] = sig_lit_AUA/sig_lit_LJ
#Tmatrix_sig[1,0] = sig_lit_LJ/sig_lit_UA
#Tmatrix_sig[1,2] = sig_lit_AUA/sig_lit_UA
#Tmatrix_sig[2,0] = sig_lit_LJ/sig_lit_AUA
#Tmatrix_sig[2,1] = sig_lit_UA/sig_lit_AUA           
           
# Initial estimates for standard deviation used in proposed distributions of MCMC
guess_var = [1,20,0.05,0.05,0.02]
# Variance (or standard deviation, need to verify which one it is) in priors for epsilon and sigma
#prior_var = [5,0.001]


#OCM: All of this first section is Rich's data setup, which I don't have any reason to alter.  I am focusing more on the monte carlo implementation




#%%



    


#%%
# Simplify notation
dnorm = distributions.norm.logpdf
dgamma = distributions.gamma.logpdf
duni = distributions.uniform.logpdf

rnorm = np.random.normal
runif = np.random.rand

norm=distributions.norm.pdf
unif=distributions.uniform.pdf

properties = 'rhol'

def calc_posterior(model,eps,sig,L,Q):

    logp = 0
#    print(eps,sig)
    # Using noninformative priors
    logp += duni(sig, 0.2, 0.5)
    logp += duni(eps, 100,200) 
    
    if model == 0:
        Q=0
    
    
    if model == 1:
        logp+=duni(Q,0,2)
        logp+=duni(L,0,1)
    # OCM: no reason to use anything but uniform priors at this point.  Could probably narrow the prior ranges a little bit to improve acceptance,
    #But Rich is rightly being conservative here especially since evaluations are cheap.
    
#    print(eps,sig)
    #rhol_hat_fake = rhol_hat_models(T_lin,model,eps,sig)
    rhol_hat = rhol_hat_models(T_rhol_data,model,eps,sig,L,Q) #[kg/m3]
    Psat_hat = Psat_hat_models(T_Psat_data,model,eps,sig,L,Q) #[kPa]        
 
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
    #return rhol_hat
    
    #OCM: Standard calculation of the log posterior. Note that t_rhol and t_Psat are precisions
    #This is one of the most important areas of the code for testing as it is where you can substitute in different data for training or change what property it is training on.

def jacobian(n_models,n_params,w,lamda,AUA_opt_params,AUA_Q_opt_params):
    jacobian=np.ones((n_models,n_models))
    jacobian[0,1]=(1/(lamda*w))*(AUA_Q_opt_params[0]*AUA_Q_opt_params[1])/(AUA_opt_params[0]*AUA_opt_params[1])
    jacobian[1,0]=w*lamda*(AUA_opt_params[0]*AUA_opt_params[1])/(AUA_Q_opt_params[0]*AUA_Q_opt_params[1])
    #jacobian[0,1]=1/(lamda*w)
    #jacobian[1,0]=w*lamda
    return jacobian
    

def transition_function(n_models,w):
    transition_function=np.ones((n_models,n_models))
    g_0_1=unif(w,0,1)
    g_1_0=1
    transition_function[0,1]=g_1_0/g_0_1
    transition_function[1,0]=g_0_1/g_1_0
    return transition_function


def gen_Tmatrix():
    ''' Generate Transition matrices based on the optimal eps, sig, Q for different models'''
    
    obj_AUA = lambda eps_sig_Q: -calc_posterior(0,eps_sig_Q[0],eps_sig_Q[1],eps_sig_Q[2],eps_sig_Q[3])
    obj_AUA_Q = lambda eps_sig_Q: -calc_posterior(1,eps_sig_Q[0],eps_sig_Q[1],eps_sig_Q[2],eps_sig_Q[3])
    
    guess_AUA = [guess_0[1],guess_0[2],guess_0[3],guess_0[4]]
    guess_AUA_Q = [guess_1[1],guess_1[2],guess_1[3],guess_1[4]]
    
    # Make sure bounds are in a reasonable range so that models behave properly
    bnd_AUA = ((0.85*guess_0[1],guess_0[1]*1.15),(0.90*guess_0[2],guess_0[2]*1.1),(0.90*guess_0[3],guess_0[3]*1.1),(0.90*guess_0[4],guess_0[4]*1.1))
    bnd_AUA_Q = ((0.85*guess_1[1],guess_1[1]*1.15),(0.9*guess_1[2],guess_1[2]*1.1),(0.9*guess_1[3],guess_1[3]*1.1),(0.90*guess_1[4],guess_1[4]*1.1))
    
    #Help debug
#    print(bnd_LJ)
#    print(bnd_UA)
#    print(bnd_AUA)
    
    
    opt_AUA = minimize(obj_AUA,guess_AUA,bounds=bnd_AUA)
    opt_AUA_Q = minimize(obj_AUA_Q,guess_AUA_Q,bounds=bnd_AUA_Q)
    #Help debug
#    print(opt_LJ)
#    print(opt_UA)
#    print(opt_AUA)
        
    AUA_opt_params = opt_AUA.x[0], opt_AUA.x[1],opt_AUA.x[2],opt_AUA.x[3]
    AUA_Q_opt_params = opt_AUA_Q.x[0], opt_AUA_Q.x[1],opt_AUA_Q.x[2],opt_AUA_Q.x[3]
    
    return AUA_opt_params, AUA_Q_opt_params

AUA_opt_params,AUA_Q_opt_params = gen_Tmatrix()

#%%

#The fraction of times a model swap is suggested as the move, rather than an intra-model move

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


guess_params=np.zeros((2,np.size(guess_0)))
guess_params[0,:]=guess_0
guess_params[1,:]=guess_1

initial_sd = [1,2, 0.01,0.01,0.5]
guess_sd=np.zeros((2,np.size(guess_0)))
guess_sd[0,:]=initial_sd
guess_sd[1,:]=initial_sd
n_models=2

def mcmc_prior_proposal(n_models,calc_posterior,guess_params,guess_sd):
    swap_freq=0.0
    n_iter=2000000
    tune_freq=100
    tune_for=10000
    parameter_prior_proposal=np.empty((n_models,np.size(guess_params,1),2))

    for i in range(1,n_models):
        initial_values=guess_params[i,:]
        initial_sd=guess_sd[i,:]
        trace,logp_trace,attempt_matrix,acceptance_matrix,prop_sd,accept_vector = RJMC_outerloop(calc_posterior,n_iter,initial_values,initial_sd,n_models,swap_freq,tune_freq,tune_for,1,1,1,1)
        trace_tuned = trace[tune_for:]
        max_ap=np.zeros(np.size(trace_tuned,1))
        map_CI=np.zeros((np.size(trace_tuned,1),2))

        for j in range(np.size(trace_tuned,1)):
            bins,values=np.histogram(trace_tuned[:,j],bins=100)
            max_ap[j]=(values[np.argmax(bins)+1]+values[np.argmax(bins)])/2
            map_CI[j]=hpd(trace_tuned[:,j],alpha=0.05)
            sigma_hat=np.sqrt(map_CI[j,1]-map_CI[j,0])/(2*1.96)
            parameter_prior_proposal[i,j]=[max_ap[j],sigma_hat*5]
            support=np.linspace(np.min(trace_tuned[:,j]),np.max(trace_tuned[:,j]),100)
            plt.hist(trace_tuned[:,j],density=True,bins=50)
            plt.plot(support,norm(support,*parameter_prior_proposal[i,j]))
            plt.show()
        plt.scatter(trace_tuned[:,3],trace_tuned[:,4])
        plt.show()
    return parameter_prior_proposal,trace_tuned





parameter_prior_proposals,trace_tuned=mcmc_prior_proposal(n_models,calc_posterior,guess_params,guess_sd)

def calc_posterior_refined(model,eps,sig,Q):

    logp = 0
#    print(eps,sig)
    # Using noninformative priors
 
    
    if model == 0:
        Q=0
        logp += duni(eps,*parameter_prior_proposals[0,1])
        logp += duni(sig,*parameter_prior_proposals[0,2])
    
    
    if model == 1:
        logp += duni(eps,*parameter_prior_proposals[1,1])
        logp += duni(sig,*parameter_prior_proposals[1,2])
        logp += duni(Q,*parameter_prior_proposals[1,3])
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
    #return rhol_hat

initial_values=guess_1 # Can use critical constants
initial_sd = [1,2, 0.005,0.01]
n_iter=1000000
tune_freq=100
tune_for=10000
n_models=2
swap_freq=0.01
#The fraction of times a model swap is suggested as the move, rather than an intra-model move
#trace,logp_trace,attempt_matrix,acceptance_matrix,prop_sd,accept_vector = RJMC_outerloop(calc_posterior_refined,n_iter,initial_values,initial_sd,n_models,swap_freq,tune_freq,tune_for,jacobian,transition_function,AUA_opt_params,AUA_Q_opt_params)


#def MCMC_priors(RJMC_outerloop)


#%%
# POST PROCESSING

print('Attempted Moves')
print(attempt_matrix)
print('Accepted Moves')
print(acceptance_matrix)
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


prob_0 = 1.*model_count[0]/(n_iter-tune_for)
print('Percent that  model 0 is sampled: '+str(prob_0 * 100.)) #The percent that use 1 parameter model

prob_1 = 1.*model_count[1]/(n_iter-tune_for)
print('Percent that model 1 is sampled: '+str(prob_1 * 100.)) #The percent that use two center UA LJ

Exp_ratio=prob_0/prob_1

#print('Analytical sampling ratio: %2.3f' % ratio)
print('Experimental sampling ratio: %2.3f' % Exp_ratio )


print('Detailed Balance')
print(prob_0*prob_matrix[0,1])
print(prob_1*prob_matrix[1,0])
    
trace_model_0=[]
trace_model_1=[]
log_trace_0=[]
log_trace_1=[]

plt.plot(logp_trace,label='Log Posterior')
plt.legend()
plt.show()

for i in range(np.size(trace_tuned,0)):
    if trace_tuned[i,0] == 0:
        trace_model_0.append(trace_tuned[i])
        log_trace_0.append(logp_trace[i])
    elif trace_tuned[i,0] == 1:
        trace_model_1.append(trace_tuned[i])
        log_trace_1.append(logp_trace[i])
        
trace_model_0=np.asarray(trace_model_0)
trace_model_1=np.asarray(trace_model_1)
f = plt.figure()
plt.scatter(trace_model_0[::10,1],trace_model_0[::10,2],s=1,label='AUA',marker=',',alpha=0.7)
plt.scatter(trace_model_1[::10,1],trace_model_1[::10,2],s=1,label='AUA+Q',marker=',',alpha=0.7)
plt.legend()
plt.xlabel(r'$\epsilon$')
plt.ylabel(r'$\sigma$')
plt.show()

#plt.hist(trace[::500,1],trace[::500,2])
#plt.show()

plt.scatter(trace_model_0[::10,1],trace_model_0[::10,3],s=1,label='AUA',marker=',',alpha=0.7)
plt.scatter(trace_model_1[::10,1],trace_model_1[::10,3],s=1,label='AUA+Q',marker=',',alpha=0.7)
plt.legend()
plt.xlabel(r'$\epsilon$')
plt.ylabel(r'Q')
plt.show()

plt.scatter(trace_model_0[::10,2],trace_model_0[::10,3],s=1,label='AUA',marker=',',alpha=0.7)
plt.scatter(trace_model_1[::10,2],trace_model_1[::10,3],s=1,label='AUA+Q',marker=',',alpha=0.7)
plt.legend()
plt.xlabel(r'$\sigma$')
plt.ylabel(r'Q')
plt.show()



plt.hist(trace_model_0[:,1],bins=50,label=r'$\epsilon$ values AUA',density=True)
plt.hist(trace_model_1[:,1],bins=50,label=r'$\epsilon$ values AUA+Q',density=True)

plt.legend()
plt.show()


plt.hist(trace_model_0[:,2],bins=50,label=r'$\sigma$ values AUA',density=True)
plt.hist(trace_model_1[:,2],bins=50,label=r'$\sigma$ values AUA+Q',density=True)
plt.legend()
plt.show()

#plt.hist(trace_model_0[:,2],bins=100,label=r'$\sigma$ values AUA',density=True)
plt.hist(trace_model_1[:,3],bins=50,label=r'Q values AUA+Q',density=True)
plt.legend()
plt.show()


'''
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
        '''