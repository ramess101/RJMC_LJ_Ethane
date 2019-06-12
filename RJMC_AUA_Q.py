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


# Here we have chosen ethane as the test case


compound='C2H2'
ff_params_ref,Tc_lit,M_w,thermo_data,NIST_bondlength=parse_data_ffs(compound)
#Retrieve force field literature values, constants, and thermo data

T_min = 0.55*Tc_lit[0]
T_max = 0.95*Tc_lit[0]
n_points=10


thermo_data=filter_thermo_data(thermo_data,T_min,T_max,10)

uncertainties=calculate_uncertainties(thermo_data,Tc_lit[0])


# Limit temperature range to that which is typical of ITIC MD simulations
thermo_data_rhoL=np.asarray(thermo_data['rhoL'])
thermo_data_Pv=np.asarray(thermo_data['Pv'])
thermo_data_SurfTens=np.asarray(thermo_data['SurfTens'])

# Substantiate LennardJones class
#Ethane_LJ = LennardJones(M_w)
compound_2CLJ = LennardJones_2C(M_w)


'''
# Epsilon and sigma can be obtained from the critical constants
#eps_Tc = Ethane_LJ.calc_eps_Tc(Tc_RP) #[K]
#sig_rhoc = Ethane_LJ.calc_sig_rhoc(rhoc_RP) #[nm]



# Create functions that return properties for a given model, eps, sig

def rhol_hat_models(Temp,model,eps,sig,L,Q):
    
    L_nm=L/10
    sig_nm=sig/10
    Q_nm=Q/10
    
    if model == 0: #Two center AUA LJ
    
        rhol_hat = compound_2CLJ.rhol_hat_2CLJQ(Temp,eps,sig,L,0) 
        
    elif model == 1: #Two center AUA LJ+Q
    
        rhol_hat = compound_2CLJ.rhol_hat_2CLJQ(Temp,eps,sig,L,Q) 
        
    return rhol_hat #[kg/m3]       
  
def Psat_hat_models(Temp,model,eps,sig,L,Q):
    
    L_nm=L/10
    sig_nm=sig/10
    Q_nm=Q/10
    
    if model == 0: #Two center AUA LJ
    
        Psat_hat = compound_2CLJ.Psat_hat_2CLJQ(Temp,eps,sig,L,0) 
        
    elif model == 1: #Two center AUA LJ+Q
    
        Psat_hat = compound_2CLJ.Psat_hat_2CLJQ(Temp,eps,sig,L,Q) 
        
    return Psat_hat #[kPa]       

def SurfTens_hat_models(Temp,model,eps,sig,L,Q):
    
    L_nm=L/10
    sig_nm=sig/10
    Q_nm=Q/10
    
    if model == 0:
        
        SurfTens_hat=compound_2CLJ.ST_hat_2CLJQ(Temp,eps,sig,L,0)
        
    elif model == 1:
        
        SurfTens_hat=compound_2CLJ.ST_hat_2CLJQ(Temp,eps,sig,L,Q)
        
    return SurfTens_hat

def T_c_hat_models(model,eps,sig,L,Q):
    
    L_nm=L/10
    sig_nm=sig/10
    Q_nm=Q/10
    
    if model == 0: 
        
        T_c_hat=compound_2CLJ.T_c_hat_2CLJQ(eps,sig,L,0)
    
    elif model == 1: 
        
        T_c_hat=compound_2CLJ.T_c_hat_2CLJQ(eps,sig,L,Q)
        
    return T_c_hat

'''

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
#guess_2 = [1,eps_lit3_AUA,sig_lit3_AUA,Lbond_lit3_AUA,Q_lit3_AUA] 

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
number_criteria = 'two'
sig_prior=[0.2,0.5]
eps_prior=[0,200]
L_prior=[0,0.3]
Q_prior=[0,2]
def calc_posterior(model,eps,sig,L,Q):

    logp = 0
#    print(eps,sig)
    # Using noninformative priors
    logp += duni(sig, *sig_prior)
    logp += duni(eps, *eps_prior) 
    logp+=duni(L,*L_prior)
    if model == 0:
        Q=0
    
    
    if model == 1:
        logp+=duni(Q,*Q_prior)
        
    # OCM: no reason to use anything but uniform priors at this point.  Could probably narrow the prior ranges a little bit to improve acceptance,
    #But Rich is rightly being conservative here especially since evaluations are cheap.
    
#    print(eps,sig)
    #rhol_hat_fake = rhol_hat_models(T_lin,model,eps,sig)
    rhol_hat = rhol_hat_models(compound_2CLJ,thermo_data_rhoL[:,0],model,eps,sig,L,Q) #[kg/m3]
    Psat_hat = Psat_hat_models(compound_2CLJ,thermo_data_Pv[:,0],model,eps,sig,L,Q) #[kPa]   
    SurfTens_hat = SurfTens_hat_models(compound_2CLJ,thermo_data_SurfTens[:,0],model,eps,sig,L,Q)     
 
    # Data likelihood
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
    
    #OCM: Standard calculation of the log posterior. Note that t_rhol and t_Psat are precisions
    #This is one of the most important areas of the code for testing as it is where you can substitute in different data for training or change what property it is training on.

def jacobian(n_models,n_params,w,lamda,AUA_opt_params,AUA_Q_opt_params):
    jacobian=np.ones((n_models,n_models))
    
    
    
    #Optimum Matching
    jacobian[0,1]=(1/(lamda*w))*(AUA_Q_opt_params[0]*AUA_Q_opt_params[1]*AUA_Q_opt_params[2])/(AUA_opt_params[0]*AUA_opt_params[1]*AUA_opt_params[2])
    jacobian[1,0]=w*lamda*(AUA_opt_params[0]*AUA_opt_params[1]*AUA_opt_params[2])/(AUA_Q_opt_params[0]*AUA_Q_opt_params[1]*AUA_Q_opt_params[2])
    
    #Direct transfer
    #jacobian[0,1]=1/(lamda*w)
    #jacobian[1,0]=w*lamda
    
    
    #jacobian[0,1]=(1/(lamda*w))*(AUA_Q_opt_params[0]*AUA_Q_opt_params[1])/(AUA_opt_params[0]*AUA_opt_params[1])
    #jacobian[1,0]=w*lamda*(AUA_opt_params[0]*AUA_opt_params[1])/(AUA_Q_opt_params[0]*AUA_Q_opt_params[1])
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

def RJMC_outerloop(compound_2CLJ, calc_posterior,n_iterations,initial_values,initial_sd,n_models,swap_freq,tune_freq,tune_for,jacobian,transition_function,AUA_opt_params,AUA_Q_opt_params):
    
    
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
        
        new_params, new_log_prob, attempt_matrix,acceptance_matrix,acceptance = RJMC_Moves(current_params,current_model,current_log_prob,n_models,swap_freq,n_params,prop_sd,attempt_matrix,acceptance_matrix,jacobian,transition_function,record_acceptance,AUA_opt_params,AUA_Q_opt_params)
        
        if acceptance == 'True':
            accept_vector[i]=1
        logp_trace[i+1] = new_log_prob
        trace[i+1] = new_params
        percent_deviation_trace[i+1]=computePercentDeviations(compound_2CLJ,thermo_data_rhoL[:,0],thermo_data_Pv[:,0],thermo_data_SurfTens[:,0],trace[i+1],thermo_data_rhoL[:,1],thermo_data_Pv[:,1],thermo_data_SurfTens[:,1],Tc_lit[0],rhol_hat_models,Psat_hat_models,SurfTens_hat_models,T_c_hat_models)
        
        if (not (i+1) % tune_freq) and (i < tune_for):
        
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
           
            
    return trace,logp_trace, percent_deviation_trace, attempt_matrix,acceptance_matrix,prop_sd,accept_vector


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
        
    lamda=5
    params[0] = proposed_model
    if proposed_model==1:
        
        
        #Optimum Matching
        params[1] = (AUA_Q_opt_params[0]/AUA_opt_params[0])*params[1]
        params[2] = (AUA_Q_opt_params[1]/AUA_opt_params[1])*params[2]
        params[3] = (AUA_Q_opt_params[2]/AUA_opt_params[2])*params[3]
        
        
        w=runif()
        
        #THIS IS IMPORTANT needs to be different depending on which direction
        
        #params[4]=w*2
        params[4] = -(1/lamda)*np.log(w)

    if proposed_model==0:
        
        #Optimum Matching
        params[1] = (AUA_opt_params[0]/AUA_Q_opt_params[0])*params[1]
        params[2] = (AUA_opt_params[1]/AUA_Q_opt_params[1])*params[2]
        params[3] = (AUA_opt_params[2]/AUA_Q_opt_params[2])*params[3]
        
        #w=params[4]/2
        w=np.exp(-lamda*params[4])
        
        
        params[4]=0
        

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
        params[4]=0
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
    n_iter=200000
    tune_freq=100
    tune_for=10000
    parameter_prior_proposal=np.empty((n_models,np.size(guess_params,1),2))

    for i in range(1,n_models):
        initial_values=guess_params[i,:]
        initial_sd=guess_sd[i,:]
        trace,logp_trace,percent_deviation_trace, attempt_matrix,acceptance_matrix,prop_sd,accept_vector = RJMC_outerloop(calc_posterior,n_iter,initial_values,initial_sd,n_models,swap_freq,tune_freq,tune_for,1,1,1,1)
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





#parameter_prior_proposals,trace_tuned=mcmc_prior_proposal(n_models,calc_posterior,guess_params,guess_sd)

def calc_posterior_refined(model,eps,sig,L,Q):

    logp = 0
#    print(eps,sig)
    # Using noninformative priors
 
    
    if model == 0:
        Q=0
        logp += dnorm(eps,*parameter_prior_proposals[0,1])
        logp += dnorm(sig,*parameter_prior_proposals[0,2])
    
    
    if model == 1:
        logp += dnorm(eps,*parameter_prior_proposals[1,1])
        logp += dnorm(sig,*parameter_prior_proposals[1,2])
        logp += dnorm(L,*parameter_prior_proposals[1,3])
        logp += dnorm(Q,*parameter_prior_proposals[1,4])
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

initial_values=guess_0 # Can use critical constants
initial_sd = np.asarray(initial_values)/100
n_iter=10000000
tune_freq=100
tune_for=10000
n_models=2
swap_freq=0.1



print('Compound: '+compound)
print('Properties: '+properties)
print('MCMC Steps: '+str(n_iter))

#The fraction of times a model swap is suggested as the move, rather than an intra-model move
trace,logp_trace,percent_deviation_trace, attempt_matrix,acceptance_matrix,prop_sd,accept_vector = RJMC_outerloop(compound_2CLJ,calc_posterior,n_iter,initial_values,initial_sd,n_models,swap_freq,tune_freq,tune_for,jacobian,transition_function,AUA_opt_params,AUA_Q_opt_params)


#def MCMC_priors(RJMC_outerloop)


#%%
# POST PROCESSING

print('Attempted Moves')
print(attempt_matrix)
print('Accepted Moves')
print(acceptance_matrix)
prob_matrix=acceptance_matrix/attempt_matrix
transition_matrix=np.ones((2,2))
transition_matrix[0,1]=acceptance_matrix[0,1]/np.sum(attempt_matrix,1)[0]
transition_matrix[1,0]=acceptance_matrix[1,0]/np.sum(attempt_matrix,1)[1]
transition_matrix[0,0]=1-transition_matrix[0,1]
transition_matrix[1,1]=1-transition_matrix[1,0]
print('Transition Matrix:')
print(transition_matrix)
trace_tuned = trace[tune_for:]
trace_tuned[:,2:]*=10
percent_deviation_trace_tuned = percent_deviation_trace[tune_for:]
model_params = trace_tuned[0,:]

fname=compound+'_'+properties+'_'+str(n_points)+'_'+str(n_iter)+'_'+str(date.today())

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
model_count = np.array([len(trace_tuned[trace_tuned[:,0]==0]),len(trace_tuned[trace_tuned[:,0]==1])])


prob_0 = 1.*model_count[0]/(n_iter-tune_for+1)
print('Percent that  model 0 is sampled: '+str(prob_0 * 100.)) #The percent that use 1 parameter model

prob_1 = 1.*model_count[1]/(n_iter-tune_for+1)
print('Percent that model 1 is sampled: '+str(prob_1 * 100.)) #The percent that use two center UA LJ

prob=[prob_0,prob_1]

Exp_ratio=prob_0/prob_1

#plot_bar_chart(prob,fname,properties,compound,n_iter,n_models)

create_percent_dev_triangle_plot(percent_deviation_trace_tuned,fname,'percent_dev_trace',new_lit_devs,prob,properties,compound,n_iter)

#print('Analytical sampling ratio: %2.3f' % ratio)
print('Experimental sampling ratio: %2.3f' % Exp_ratio )


print('Detailed Balance')
print(prob_0*transition_matrix[0,1])
print(prob_1*transition_matrix[1,0])


    
trace_model_0=[]
trace_model_1=[]
log_trace_0=[]
log_trace_1=[]


plt.plot(logp_trace,label='Log Posterior')
plt.legend()
plt.show()

np.save('trace/trace_'+fname+'.npy',trace_tuned)
np.save('logprob/logprob_'+fname+'.npy',logp_trace)
np.save('percent_dev/percent_dev_'+fname+'.npy',percent_deviation_trace_tuned)



for i in range(np.size(trace_tuned,0)):
    if trace_tuned[i,0] == 0:
        trace_model_0.append(trace_tuned[i])
        log_trace_0.append(logp_trace[i])
    elif trace_tuned[i,0] == 1:
        trace_model_1.append(trace_tuned[i])
        log_trace_1.append(logp_trace[i])
        
        
        
trace_model_0=np.asarray(trace_model_0)
trace_model_1=np.asarray(trace_model_1)



create_param_triangle_plot_4D(trace_model_0,fname,'trace_model_0',lit_params,properties,compound,n_iter)
create_param_triangle_plot_4D(trace_model_1,fname,'trace_model_1',lit_params,properties,compound,n_iter)

get_metadata(compound,properties,sig_prior,eps_prior,L_prior,Q_prior,n_iter,swap_freq,n_points,transition_matrix,prob,attempt_matrix,acceptance_matrix)
#%%
'''
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
