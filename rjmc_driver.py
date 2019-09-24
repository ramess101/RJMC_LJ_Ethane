#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 16:15:33 2019

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
from RJMC_auxiliary_functions import *
from datetime import date
import copy
from pymbar import BAR,timeseries
import random
import sys
from RJMC_2CLJ_AUA_Q import *

compound='C2H6'
properties=['rhol+Psat','two']
temp_range=[0.55,0.95]
n_points=10
prior_type=['exponential','exponential']
eps_sig_L_prior_params=[40,40,40]
Q_prior=[0,1]
n_iter=5*10**6
prior_max_values=[400,5,3,1]

trace,logp_trace,percent_deviation_trace, attempt_matrix,acceptance_matrix,prop_sd,accept_vector,alpha_vector=run_RJMC(compound,properties,temp_range,n_points,eps_sig_L_prior_params,Q_prior,prior_type,n_iter,biasing_factor_AUA=0,biasing_factor_UA=0,swap_freq=0.1,initial_model='AUA+Q',prior_max_values=prior_max_values,optimum_matching='True')



#%%
# POST PROCESSING
tune_for=100000
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
percent_deviation_trace_tuned = percent_deviation_trace[tune_for:]

#[t0,g,Neff_max]=timeseries.detectEquilibration(logp_trace_tuned,nskip=1000)

trace_equil=trace_tuned
logp_trace_equil=logp_trace_tuned

trace_equil[:,2:]*=10
percent_deviation_trace_equil = percent_deviation_trace[tune_for:]
model_params = trace_equil[0,:]

#[t0,g,Neff_max]=timeseries.detectEquilibration(logp_trace_tuned,nskip=100)

#logp_trace_equil=logp_trace_tuned[t0:]
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

create_percent_dev_triangle_plot(percent_deviation_trace_tuned,fname,'percent_dev_trace',lit_devs,prob,properties,compound,n_iter)

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

#np.save('trace_'+fname+'.npy',trace_tuned)
#np.save('logprob/logprob_'+fname+'.npy',logp_trace)
#np.save('percent_dev/percent_dev_'+fname+'.npy',percent_deviation_trace_tuned)
#Save trajectories (can be disabled since they are big files)


for i in range(np.size(trace_equil,0)):
    if trace_equil[i,0] == 0:
        trace_model_0.append(trace_equil[i])
        #log_trace_0.append(logp_trace[i])
    elif trace_equil[i,0] == 1:
        trace_model_1.append(trace_equil[i])
        #log_trace_1.append(logp_trace[i])
    elif trace_equil[i,0] == 2:
        trace_model_2.append(trace_equil[i])
        #log_trace_2.append(logp_trace[i])        
        
        
trace_model_0=np.asarray(trace_model_0)
trace_model_1=np.asarray(trace_model_1)
trace_model_2=np.asarray(trace_model_2)

#plt.hist(alpha_vector[0],range=[-10,100],bins=50,alpha=0.7)
#plt.hist(alpha_vector[1],range=[-10,100],bins=50,alpha=0.7)
#plt.show()
#sys.exit(np.mean(logp_trace_tuned))
#create_param_triangle_plot_4D(trace_model_0,fname,'trace_model_0',lit_params,properties,compound,n_iter)
#create_param_triangle_plot_4D(trace_model_1,fname,'trace_model_1',lit_params,properties,compound,n_iter)
#create_param_triangle_plot_4D(trace_model_2,fname,'trace_model_2',lit_params,properties,compound,n_iter)

#Plot parameters

#get_metadata(directory,label,compound,properties,sig_prior,eps_prior,L_prior,Q_prior,n_iter,swap_freq,n_points,transition_matrix,prob,attempt_matrix,acceptance_matrix)

#write outputs to file
