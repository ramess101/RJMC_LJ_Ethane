#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 14:55:02 2019

@author: owenmadin
"""

import yaml
from datetime import date
import os

simulation_params = {}

#BASIC PARAMS
simulation_params['compound'] = 'C2H4'
# Compound to use (C2H2,C2H4,C2H6,C2F4,O2,N2,Br2,F2)
simulation_params['properties'] = 'rhol+Psat'
# Which properties to simulate ('rhol','rhol+Psat','All')
simulation_params['trange'] = [0.55,0.95]
#Temperature range (fraction of Tc)
simulation_params['steps'] = 2000000
#Number of MCMC steps
simulation_params['swap_freq'] = 0.1
#Frequency of model swaps
simulation_params['number_data_points'] = 10

#CUSTOM SIMULATION OPTIONS
simulation_params['priors'] = {'epsilon': ['exponential', [0,400]],
        'sigma': ['exponential', [0,5]],
        'L': ['exponential', [0,3]],
        'Q': ['exponential', [0,1]]}
#Options: exponential [loc (should always be 0), scale]
#         gamma [alpha,beta,loc,scale]
#See scipy.stats.expon and scipy.stats.gengamma

simulation_params['optimum_matching'] = ['True','True']
#First value is for AUA/AUAQ maps, second is for AUA/UA

simulation_params['biasing_factor'] = [0,0,0]
#Corresponds to [AUA,AUAQ,UA]

simulation_params['simulation_type'] = 'Refit_Prior'
#'Basic', 'Refit_Prior' or 'Biasing_Factor'

simulation_params['opt_bounds'] = 'Expanded'
#Whether to use tight optimization bounds or expanded bounds for calculating
#AUA-AUAQ transitions.  Overwritten if use_MAP is true.

#RECORDING OPTIONS
simulation_params['save_traj'] = True
#Saves trajectories
simulation_params['label'] = 'production'
#Label for output files
simulation_params['USE_BAR'] = True
#Whether to use BAR to calculate Bayes Factors


#REFIT PRIOR
if simulation_params['simulation_type'] == 'Refit_Prior':
    simulation_params['single_simulation_length'] = 100000
    #Length of single MCMC simulation to refit Q prior (probably want 1000000 max)
    simulation_params['refit_prior_to'] = 'exponential'
    #What prior will we fit Q to?  exponential or gamma (gamma currently unreliable)
    simulation_params['use_MAP'] = False
    #Should we use previous simulations to create a MAP (maximum a posteriori) map
    if simulation_params['use_MAP'] == True:
        simulation_params['MAP_simulations'] = ['output/C2H2/rhol+Psat/C2H2_rhol+Psat_2000000_aua_only_2019-10-25/trace/trace.npy',
         'output/C2H2/rhol+Psat/C2H2_rhol+Psat_2000000_auaq_only_2019-10-25/trace/trace.npy']
        # Simulations for MAP finding [AUA,AUAQ]

#BIASING FACTOR
if simulation_params['simulation_type'] == 'Biasing_Factor':
    simulation_params['biasing_factor_simulation_length'] = 100000
    #Length of 3 single MCMC simulations to determine biasing factors
    simulation_params['refit_prior'] = True
    #Should we do a prior refit
    if simulation_params['refit_prior'] is True:
        simulation_params['single_simulation_length'] = 100000
        simulation_params['refit_prior_to'] = 'exponential'
    simulation_params['use_MAP'] = False
    if simulation_params['use_MAP'] == True:
        simulation_params['MAP_simulations'] = ['output/C2H2/rhol+Psat/C2H2_rhol+Psat_2000000_aua_only_2019-10-25/trace/trace.npy',
         'output/C2H2/rhol+Psat/C2H2_rhol+Psat_2000000_auaq_only_2019-10-25/trace/trace.npy']


#label
today = str(date.today())

if os.path.exists('runfiles') is False:
    os.mkdir('runfiles')

filename = 'runfiles/'+simulation_params['compound'] + '_'+simulation_params['properties']+'_'+simulation_params['simulation_type']+'_'+simulation_params['label']+'_'+today+'.yml' 


with open(filename,'w') as outfile:
    yaml.dump(simulation_params,outfile,default_flow_style=False)
