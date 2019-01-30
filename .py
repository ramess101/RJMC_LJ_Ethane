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

# Define probabilities for both regions

def region1(x,y)