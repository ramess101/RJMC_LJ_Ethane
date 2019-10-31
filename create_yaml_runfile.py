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
simulation_params['compound'] = 'C2H6'
simulation_params['properties'] = 'rhol+Psat'
simulation_params['trange'] = [0.55,0.95]
simulation_params['steps'] = 20000
simulation_params['swap_freq'] = 0.1
simulation_params['number_data_points'] = 10

#CUSTOM SIMULATION OPTIONS
simulation_params['priors'] = {'epsilon': ['exponential', [0,400]],
        'sigma': ['exponential', [0,5]],
        'L': ['exponential', [0,3]],
        'Q': ['exponential', [0,1]]}
simulation_params['optimum_matching'] = ['True','True']
simulation_params['biasing_factor'] = [0,0,0]
simulation_params['simulation_type'] = 'Biasing_Factor'

#RECORDING OPTIONS
simulation_params['save_traj'] = True
simulation_params['label'] = 'YAML_TEST'
simulation_params['USE_BAR'] = True

#REFIT PRIOR
if simulation_params['simulation_type'] == 'Refit_Prior':
    simulation_params['single_simulation_length'] = 100000
    simulation_params['refit_prior_to'] = 'exponential'
    simulation_params['use_MAP'] = False
    if simulation_params['use_MAP'] == True:
        simulation_params['MAP_simulations'] = ['output/C2H2/rhol+Psat/C2H2_rhol+Psat_2000000_aua_only_2019-10-25/trace/trace.npy',
         'output/C2H2/rhol+Psat/C2H2_rhol+Psat_2000000_auaq_only_2019-10-25/trace/trace.npy']

#BIASING FACTOR
if simulation_params['simulation_type'] == 'Biasing_Factor':
    simulation_params['biasing_factor_simulation_length'] = 20000
    simulation_params['refit_prior'] = True
    if simulation_params['refit_prior'] is True:
        simulation_params['single_simulation_length'] = 20000
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
