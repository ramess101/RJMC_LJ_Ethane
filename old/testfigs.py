#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 09:56:19 2018

@author: owenmadin
"""

import csv
import pymbar
from pymbar import testsystems, MBAR, timeseries
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import copy
import os
import os.path
import optparse
import scipy
from optparse import OptionParser
from scipy.stats import distributions


normal1=distributions.norm(3,1.1)
normal2=distributions.norm(7,1)
uniform=distributions.uniform(0,10)

gamma=distributions.gamma(6,0.01)
x=np.linspace(0,10,100)
y1=normal1.pdf(x)+normal2.pdf(x)
y2=uniform.pdf(x)

plt.plot(x,y2)
plt.fill_between(x,0,y2)
plt.title('Prior Distribution')
plt.ylim([0,0.5])
plt.show()


plt.plot(x,y1*y2)
#plt.plot(x,gamma.pdf(x))
plt.fill_between(x,0,y1*y2)
plt.title('Posterior Distribution')
plt.show()

normal2=distributions.norm(7,2)
y1=normal1.pdf(x)+normal2.pdf(x)

plt.plot(x,y1*y2)
plt.fill_between(x,0,y1*y2)
plt.title('New Posterior Distribution with additional data')

plt.show()