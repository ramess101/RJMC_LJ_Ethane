#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 14:42:37 2019

@author: owenmadin
"""
from __future__ import division
import numpy as np
import argparse
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
from pymbar import BAR, timeseries
import random
import sys
from RJMC_2CLJQ_OOP import RJMC_Simulation, RJMC_Prior
from biasing_factor_simulation import compute_biasing_factors


def parse_input_yaml(filepath):
    print('Loading simulation params from '+filepath+'...')
    with open(filepath) as yfile:
        simulation_params = yaml.load(yfile)#,Loader=yaml.FullLoader)
    return simulation_params

def basic(simulation_params):

    print(simulation_params['priors'])    
    prior = RJMC_Prior(simulation_params['priors'])
    prior.epsilon_prior()
    prior.sigma_prior()
    prior.L_prior()
    prior.Q_prior()

    rjmc_simulator = RJMC_Simulation(simulation_params['compound'], 
                                     simulation_params['trange'],
                                     simulation_params['properties'],
                                     simulation_params['number_data_points'],
                                     simulation_params['steps'],
                                     simulation_params['swap_freq'],
                                     simulation_params['biasing_factor'],
                                     simulation_params['optimum_matching'])

    rjmc_simulator.prepare_data()

    print('Simulation Attributes:', rjmc_simulator.get_attributes())

    compound_2CLJ = LennardJones_2C(rjmc_simulator.M_w)
    rjmc_simulator.optimum_bounds = simulation_params['opt_bounds']
    rjmc_simulator.gen_Tmatrix(prior, compound_2CLJ)
    # print(rjmc_simulator.opt_params_AUA)
    rjmc_simulator.set_initial_state(prior, compound_2CLJ)

    rjmc_simulator.RJMC_Outerloop(prior, compound_2CLJ)
    trace, logp_trace, percent_dev_trace, BAR_trace = rjmc_simulator.Report(USE_BAR=simulation_params['USE_BAR'])
    
    rjmc_simulator.write_output(simulation_params['priors'], tag=simulation_params['label'], save_traj=simulation_params['save_traj'])

    path = 'output/' + simulation_params['compound'] + '/' + simulation_params['properties'] + '/' + simulation_params['compound'] + \
            '_' + simulation_params['properties'] + '_' + str(simulation_params['steps']) + '_' + simulation_params['label'] + '_' + str(date.today()) +'/runfile.yaml'

    
    with open(path,'w') as outfile:
        yaml.dump(simulation_params,outfile,default_flow_style=False)
    
def refit_prior(simulation_params):

    prior = RJMC_Prior(simulation_params['priors'])
    prior.epsilon_prior()
    prior.sigma_prior()
    prior.L_prior()
    prior.Q_prior()

    print('Approximating AUA+Q posterior distribution')

    mcmc_prior_simulation = RJMC_Simulation(simulation_params['compound'], 
                                     simulation_params['trange'],
                                     simulation_params['properties'],
                                     simulation_params['number_data_points'],
                                     simulation_params['single_simulation_length'],
                                     0.0,
                                     [0,0,0],
                                     simulation_params['optimum_matching'])
    mcmc_prior_simulation.prepare_data()
    compound_2CLJ = LennardJones_2C(mcmc_prior_simulation.M_w)
    mcmc_prior_simulation.optimum_bounds = simulation_params['opt_bounds']
    mcmc_prior_simulation.gen_Tmatrix(prior, compound_2CLJ)
    mcmc_prior_simulation.set_initial_state(prior,
                                            compound_2CLJ,
                                            initial_model='AUA+Q')
    mcmc_prior_simulation.RJMC_Outerloop(prior, compound_2CLJ)
    mcmc_prior_simulation.Report()
    simulation_params['priors']['Q'] = mcmc_prior_simulation.refit_prior(simulation_params['refit_prior_to'])
    print('Refitting Q prior')
    print(simulation_params['priors']['Q'])

    prior = RJMC_Prior(simulation_params['priors'])
    prior.epsilon_prior()
    prior.sigma_prior()
    prior.L_prior()
    prior.Q_prior()
    
    if simulation_params['use_MAP'] is True:
        aua_max,auaq_max = create_map(simulation_params['MAP_simulations'][0],simulation_params['MAP_simulations'][1])
    
    rjmc_simulator = RJMC_Simulation(simulation_params['compound'], 
                                     simulation_params['trange'],
                                     simulation_params['properties'],
                                     simulation_params['number_data_points'],
                                     simulation_params['steps'],
                                     simulation_params['swap_freq'],
                                     simulation_params['biasing_factor'],
                                     simulation_params['optimum_matching'])
    rjmc_simulator.prepare_data()

    print('Simulation Attributes:', rjmc_simulator.get_attributes())


    compound_2CLJ = LennardJones_2C(rjmc_simulator.M_w)
    rjmc_simulator.optimum_bounds = simulation_params['opt_bounds']
    rjmc_simulator.gen_Tmatrix(prior, compound_2CLJ)
    rjmc_simulator.set_initial_state(prior, compound_2CLJ)

    if simulation_params['use_MAP'] is True:
         custom_map = list([list(aua_max),list(auaq_max),list(rjmc_simulator.opt_params_UA)])
         rjmc_simulator.opt_params_AUA,rjmc_simulator.opt_params_AUA_Q,rjmc_simulator.opt_params_UA = rjmc_simulator.load_custom_map(custom_map)

    rjmc_simulator.RJMC_Outerloop(prior, compound_2CLJ)
    trace, logp_trace, percent_dev_trace,BAR_trace = rjmc_simulator.Report(USE_BAR=simulation_params['USE_BAR'])
    rjmc_simulator.write_output(simulation_params['priors'], tag=simulation_params['label'], save_traj=simulation_params['save_traj'])
    
    path = 'output/' + simulation_params['compound'] + '/' + simulation_params['properties'] + '/' + simulation_params['compound'] + \
            '_' + simulation_params['properties'] + '_' + str(simulation_params['steps']) + '_' + simulation_params['label'] + '_' + str(date.today()) +'/runfile.yaml'

    with open(path,'w') as outfile:
        yaml.dump(simulation_params,outfile,default_flow_style=False)
    
def biasing_factor(simulation_params):
    
    if simulation_params['refit_prior'] is True:
        prior = RJMC_Prior(simulation_params['priors'])
        prior.epsilon_prior()
        prior.sigma_prior()
        prior.L_prior()
        prior.Q_prior()
    
        print('Approximating AUA+Q posterior distribution')
    
        mcmc_prior_simulation = RJMC_Simulation(simulation_params['compound'], 
                                         simulation_params['trange'],
                                         simulation_params['properties'],
                                         simulation_params['number_data_points'],
                                         simulation_params['single_simulation_length'],
                                         0.0,
                                         [0,0,0],
                                         simulation_params['optimum_matching'])
        mcmc_prior_simulation.prepare_data()
        compound_2CLJ = LennardJones_2C(mcmc_prior_simulation.M_w)
        mcmc_prior_simulation.gen_Tmatrix(prior, compound_2CLJ)
        mcmc_prior_simulation.set_initial_state(prior,
                                                compound_2CLJ,
                                                initial_model='AUA+Q')
        mcmc_prior_simulation.RJMC_Outerloop(prior, compound_2CLJ)
        mcmc_prior_simulation.Report()
        simulation_params['priors']['Q'] = mcmc_prior_simulation.refit_prior(simulation_params['refit_prior_to'])
        print('Refitting Q prior')
        print(simulation_params['priors']['Q'])
    
    print('Starting AUA')
    AUA_simulation = RJMC_Simulation(simulation_params['compound'], 
                                         simulation_params['trange'],
                                         simulation_params['properties'],
                                         simulation_params['number_data_points'],
                                         simulation_params['biasing_factor_simulation_length'],
                                         0.0,
                                         [0,0,0],
                                         simulation_params['optimum_matching'])
    AUA_simulation.prepare_data()
    compound_2CLJ = LennardJones_2C(AUA_simulation.M_w)
    AUA_simulation.gen_Tmatrix(prior, compound_2CLJ)
    AUA_simulation.set_initial_state(prior,
                                     compound_2CLJ,
                                     initial_model='AUA')
    AUA_simulation.RJMC_Outerloop(prior, compound_2CLJ)
    AUA_simulation.Report()
    AUA_logp_trace=AUA_simulation.logp_trace
    print('AUA Complete!')
    print('Starting AUA+Q')
    AUAQ_simulation = RJMC_Simulation(simulation_params['compound'], 
                                         simulation_params['trange'],
                                         simulation_params['properties'],
                                         simulation_params['number_data_points'],
                                         simulation_params['biasing_factor_simulation_length'],
                                         0.0,
                                         [0,0,0],
                                         simulation_params['optimum_matching'])
    AUAQ_simulation.prepare_data()
    compound_2CLJ = LennardJones_2C(AUAQ_simulation.M_w)
    AUAQ_simulation.gen_Tmatrix(prior, compound_2CLJ)
    AUAQ_simulation.set_initial_state(prior,
                                     compound_2CLJ,
                                     initial_model='AUA+Q')
    AUAQ_simulation.RJMC_Outerloop(prior, compound_2CLJ)
    AUAQ_simulation.Report()
    AUAQ_logp_trace=AUAQ_simulation.logp_trace    
    print('AUA Complete!')
    
    print('Starting UA')
    UA_simulation = RJMC_Simulation(simulation_params['compound'], 
                                         simulation_params['trange'],
                                         simulation_params['properties'],
                                         simulation_params['number_data_points'],
                                         simulation_params['biasing_factor_simulation_length'],
                                         0.0,
                                         [0,0,0],
                                         simulation_params['optimum_matching'])
    UA_simulation.prepare_data()
    compound_2CLJ = LennardJones_2C(UA_simulation.M_w)
    UA_simulation.gen_Tmatrix(prior, compound_2CLJ)
    UA_simulation.set_initial_state(prior,
                                     compound_2CLJ,
                                     initial_model='UA')
    UA_simulation.RJMC_Outerloop(prior, compound_2CLJ)
    UA_simulation.Report()
    UA_logp_trace=UA_simulation.logp_trace
    print('UA Complete!')
    
    UA_biasing_factor,AUAQ_biasing_factor=compute_biasing_factors(UA_logp_trace, AUA_logp_trace, AUAQ_logp_trace)
    
    
    biasing_factor=np.asarray([0, AUAQ_biasing_factor, UA_biasing_factor])
    print('Biasing factor', biasing_factor)
    

    rjmc_simulator = RJMC_Simulation(simulation_params['compound'], 
                                     simulation_params['trange'],
                                     simulation_params['properties'],
                                     simulation_params['number_data_points'],
                                     simulation_params['steps'],
                                     simulation_params['swap_freq'],
                                     biasing_factor,
                                     simulation_params['optimum_matching'])

    rjmc_simulator.prepare_data()

    print('Simulation Attributes:', rjmc_simulator.get_attributes())

    compound_2CLJ = LennardJones_2C(rjmc_simulator.M_w)
    rjmc_simulator.optimum_bounds = simulation_params['opt_bounds']
    rjmc_simulator.gen_Tmatrix(prior, compound_2CLJ)
    # print(rjmc_simulator.opt_params_AUA)
    rjmc_simulator.set_initial_state(prior, compound_2CLJ)

    rjmc_simulator.RJMC_Outerloop(prior, compound_2CLJ)
    trace, logp_trace, percent_dev_trace, BAR_trace = rjmc_simulator.Report(USE_BAR=simulation_params['USE_BAR'])
    
    rjmc_simulator.write_output(simulation_params['priors'], tag=simulation_params['label'], save_traj=simulation_params['save_traj'])

    path = 'output/' + simulation_params['compound'] + '/' + simulation_params['properties'] + '/' + simulation_params['compound'] + \
            '_' + simulation_params['properties'] + '_' + str(simulation_params['steps']) + '_' + simulation_params['label'] + '_' + str(date.today()) +'/runfile.yaml'

    with open(path,'w') as outfile:
        yaml.dump(simulation_params,outfile,default_flow_style=False)
    
def main():
    parser=argparse.ArgumentParser(description='Find YAML file')
    
    parser.add_argument('--filepath', '-f',
                    type=str,
                    help='',
                    required=True)

    args = parser.parse_args()
    filepath = args.filepath
    print('Parsing simulation params')
    simulation_params = parse_input_yaml(filepath)
    
    
    if simulation_params['simulation_type'] == 'Basic':
        basic(simulation_params)
    elif simulation_params['simulation_type'] == 'Refit_Prior':
        refit_prior(simulation_params)
    elif simulation_params['simulation_type'] == 'Biasing_Factor':
        biasing_factor(simulation_params)
        
    print('Finished!')   

if __name__ == '__main__':
    main()
