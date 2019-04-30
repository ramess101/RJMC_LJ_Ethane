import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy.special import logsumexp
fnames = glob('smc/*.npz')
fname = fnames[0]

from collections import defaultdict
log_Z_estimates = defaultdict(lambda : np.zeros(3))

for fname in fnames:
    print(fname)
    result = np.load(fname)
    model = int(result['model'])
    property_subset_name = str(result['property_subset'])

    forward_works = - result['current_log_weights'][-1]
    log_Z_estimate = (logsumexp(- forward_works) - np.log(len(forward_works)))
    log_Z_estimates[property_subset_name][model] = log_Z_estimate
    print('log_Z_estimate: {:.5f}'.format(log_Z_estimate))

for p in log_Z_estimates:
    print('properties: ', p)
    print('\tlog Z estimates for models [0,1,2]: ', log_Z_estimates[p])
    print('\tBayes factor estimates for models [0,1] relative to model 2: ', np.exp(log_Z_estimates[p][:2] - log_Z_estimates[p][-1]))

