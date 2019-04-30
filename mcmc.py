import numpy as np
from tqdm import tqdm


def random_walk_mh(x0, log_prob_fun, n_steps=1000, stepsize=0.1, progress_bar=True):
    """Random-walk Metropolis-Hastings with Gaussian proposals.

    Parameters
    ----------
    x0 : array of floats (dim := len(x0))
        initial state of sampler
    log_prob_fun : callable, accepts an array and returns a float
        unnormalized log probability density function
    n_steps : integer
        number of MCMC steps
    stepsize : float
        standard deviation of random walk proposal distribution

    Returns
    -------
    traj : [n_steps + 1 x dim] array of floats
        trajectory of samples generated by MCMC
    log_probs : [n_steps + 1] array of floats
        unnormalized log-probabilities of the samples
    acceptance_fraction : float in [0,1]
        fraction of accepted proposals
    """
    dim = len(x0)

    traj = [x0]
    log_probs = [log_prob_fun(x0)]

    acceptances = 0
    r = range(n_steps)
    if progress_bar:
        range_ = tqdm(r)
    else:
        range_ = r

    for n in range_:

        x_proposal = traj[-1] + stepsize * np.random.randn(dim)
        log_prob_proposal = log_prob_fun(x_proposal)

        if np.random.rand() < np.exp(log_prob_proposal - log_probs[-1]):
            traj.append(x_proposal)
            log_probs.append(log_prob_proposal)
            acceptances += 1
        else:
            traj.append(traj[-1])
            log_probs.append(log_probs[-1])

        if progress_bar:
            range_.set_postfix({'log_prob': log_probs[-1], 'accept_fraction': float(acceptances) / (1 + n)})
    del (range_)

    return np.array(traj), np.array(log_probs), float(acceptances) / n_steps
