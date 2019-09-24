#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 17:16:05 2019

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


def main():
    compound = 'O2'
    properties = 'All'
    T_range = [0.55, 0.95]
    n_points = 10
    swap_freq = 0.1
    steps = 1 * 10**6
    biasing_factor = [0, 0, 0]
    optimum_matching = ['False', 'True']

    prior_values = {
        'epsilon': ['exponential', [400]],
        'sigma': ['exponential', [5]],
        'L': ['exponential', [3]],
        'Q': ['exponential', [1]]}

    prior = RJMC_Prior(prior_values)

    prior.epsilon_prior()
    prior.sigma_prior()
    prior.L_prior()
    prior.Q_prior()

    rjmc_simulator = RJMC_Simulation(
        compound,
        T_range,
        properties,
        n_points,
        steps,
        swap_freq,
        biasing_factor,
        optimum_matching)

    rjmc_simulator.prepare_data()
    print(rjmc_simulator.get_attributes())

    compound_2CLJ = LennardJones_2C(rjmc_simulator.M_w)

    rjmc_simulator.gen_Tmatrix(prior, compound_2CLJ)
    # print(rjmc_simulator.opt_params_AUA)
    rjmc_simulator.set_initial_state(prior, compound_2CLJ)

    rjmc_simulator.RJMC_Outerloop(prior, compound_2CLJ)
    trace, logp_trace, percent_dev_trace, BAR_trace = rjmc_simulator.Report(USE_BAR=True)
    rjmc_simulator.write_output(prior_values, tag='BAR_testing', save_traj=True)
    print('Finished!')
    return trace, logp_trace, percent_dev_trace,BAR_trace


if __name__ == '__main__':
    trace, logp_trace, percent_dev_trace, BAR_trace = main()
