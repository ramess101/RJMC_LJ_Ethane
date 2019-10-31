#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 09:25:17 2019

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
from scipy.stats import expon,gengamma
import random as rm
from pymc3.stats import hpd
from RJMC_auxiliary_functions import *
from datetime import date
import copy
from pymbar import BAR, timeseries
import random
import sys
import os
from shutil import rmtree
from RJMC_2CLJQ_OOP import RJMC_Prior, RJMC_Simulation


def main():

    parser = argparse.ArgumentParser(description='Process input parameters for RJMC 2CLJQ')

    parser.add_argument('--compound', '-c',
                        type=str,
                        help='Compound to simulate parameters for',
                        required=True)

    parser.add_argument('--trange', '-t',
                        type=list,
                        help='Temperature range for data to include',
                        required=False)

    parser.add_argument('--steps', '-s',
                        type=int,
                        help='Number of MCMC steps',
                        required=True)

    parser.add_argument('--properties', '-p',
                        type=str,
                        help='Properties to include in computation',
                        required=True)

    parser.add_argument('--priors', '-r',
                        type=dict,
                        help='Values and types of prior distribution',
                        required=False)

    parser.add_argument('--optimum_matching', '-o',
                        type=list,
                        help='Whether to use optimum matching in transitions',
                        required=False)

    parser.add_argument('--number_data_points', '-d',
                        type=int,
                        help='Number of data points to include',
                        required=False)

    parser.add_argument('--swap_freq', '-f',
                        type=float,
                        help='Frequency of model jump proposal',
                        required=False)

    parser.add_argument('--biasing_factor', '-b',
                        type=str,
                        help='Biasing factors for each model',
                        required=False)
    parser.add_argument('--label', '-l',
                        type=str,
                        help='label for files',
                        required=False)
    parser.add_argument('--save_traj', '-j',
                        type=bool,
                        help='Whether to save trajectory files',
                        required=False)

    T_range = [0.55, 0.95]
    n_points = 10
    swap_freq = 0.1
    biasing_factor = [0, 0, 0]
    optimum_matching = ['True', 'True']
    tag = ''
    save_traj = False

    prior_values = {
        'epsilon': ['exponential', [0,400]],
        'sigma': ['exponential', [0,5]],
        'L': ['exponential', [0,3]],
        'Q': ['exponential', [0,1]]}

    args = parser.parse_args()
    print(args.compound)
    print(args.biasing_factor)
    
    


    compound = args.compound
    # T_range=args.trange
    steps = args.steps
    properties = args.properties

    if args.trange is not None:
        T_range = args.trange
    if args.number_data_points is not None:
        n_points = args.number_data_points
    if args.swap_freq is not None:
        swap_freq = args.swap_freq
    if args.biasing_factor is not None:
        biasing_factor_raw = list(args.biasing_factor.split(','))
        biasing_factor = [float(i) for i in biasing_factor_raw]
    if args.optimum_matching is not None:
        optimum_matching = args.optimum_matching
    if args.priors is not None:
        prior_values = args.priors
    if args.label is not None:
        tag = args.label
    if args.save_traj is not None:
        save_traj = args.save_traj

    prior = RJMC_Prior(prior_values)
    prior.epsilon_prior()
    prior.sigma_prior()
    prior.L_prior()
    prior.Q_prior()

    mcmc_prior_simulation = RJMC_Simulation(compound,
                                            T_range,
                                            properties,
                                            n_points,
                                            5 * 10**5,
                                            0.0,
                                            biasing_factor,
                                            optimum_matching)
    mcmc_prior_simulation.prepare_data()
    compound_2CLJ = LennardJones_2C(mcmc_prior_simulation.M_w)
    mcmc_prior_simulation.gen_Tmatrix(prior, compound_2CLJ)
    mcmc_prior_simulation.set_initial_state(prior,
                                            compound_2CLJ,
                                            initial_model='AUA+Q')
    mcmc_prior_simulation.RJMC_Outerloop(prior, compound_2CLJ)
    mcmc_prior_simulation.Report()
    prior_values['Q'] = mcmc_prior_simulation.refit_prior('exponential')
    print(prior_values['Q'])
    #plt.hist(mcmc_prior_simulation.trace_model_1[:,4],bins=50,density=True)
    #plt.plot(np.linspace(0,1,num=500),gengamma.pdf(np.linspace(0,1,num=500),*prior_values['Q'][1]))
    #plt.plot(np.linspace(0,1,num=500),expon.pdf(np.linspace(0,1,num=500),0,400))
    #plt.show()

    print('Refitting Prior for Q')

    prior = RJMC_Prior(prior_values)
    prior.epsilon_prior()
    prior.sigma_prior()
    prior.L_prior()
    prior.Q_prior()
    
    print(prior.Q_prior_values)
    '''
    plt.hist(mcmc_prior_simulation.trace_model_1[:,4],bins=50,density=True)
    plt.plot(np.linspace(0,1,num=500),expon.pdf(np.linspace(0,1,num=500),*prior.Q_prior_values))
    plt.show()
    '''
    
    
    aua_path = 'output/C2H2/rhol+Psat/C2H2_rhol+Psat_2000000_aua_only_2019-10-25/trace/trace.npy'
    auaq_path = 'output/C2H2/rhol+Psat/C2H2_rhol+Psat_2000000_auaq_only_2019-10-25/trace/trace.npy'
    
    aua_max,auaq_max = create_map(aua_path,auaq_path)
    
    rjmc_simulator = RJMC_Simulation(compound,
                                     T_range,
                                     properties,
                                     n_points,
                                     steps,
                                     swap_freq,
                                     biasing_factor,
                                     optimum_matching)

    rjmc_simulator.prepare_data()

    print('Simulation Attributes:', rjmc_simulator.get_attributes())


    compound_2CLJ = LennardJones_2C(rjmc_simulator.M_w)

    rjmc_simulator.gen_Tmatrix(prior, compound_2CLJ)
    rjmc_simulator.set_initial_state(prior, compound_2CLJ)
    
    custom_map = list([list(aua_max),list(auaq_max),list(rjmc_simulator.opt_params_UA)])

    
    rjmc_simulator.opt_params_AUA,rjmc_simulator.opt_params_AUA_Q,rjmc_simulator.opt_params_UA = rjmc_simulator.load_custom_map(custom_map)
    rjmc_simulator.RJMC_Outerloop(prior, compound_2CLJ)
    trace, logp_trace, percent_dev_trace,BAR_trace = rjmc_simulator.Report(USE_BAR=True)
    rjmc_simulator.write_output(prior_values, tag=tag, save_traj=save_traj)
    print('Finished!')


if __name__ == '__main__':
    main()
