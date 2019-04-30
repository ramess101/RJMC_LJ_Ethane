from RJMC_2CLJ_AUA_Q import likelihood_log_pdf_model_0, likelihood_log_pdf_model_1, likelihood_log_pdf_model_2
from RJMC_2CLJ_AUA_Q import prior_log_pdf_model_0, prior_log_pdf_model_1, prior_log_pdf_model_2
from RJMC_2CLJ_AUA_Q import sample_from_prior_model_0, sample_from_prior_model_1, sample_from_prior_model_2

models = [0, 1, 2]

prior_samplers = [
    sample_from_prior_model_0,
    sample_from_prior_model_1,
    sample_from_prior_model_2,
]

prior_log_pdfs = [
    prior_log_pdf_model_0,
    prior_log_pdf_model_1,
    prior_log_pdf_model_2,
]

likelihood_log_pdfs = [
    likelihood_log_pdf_model_0,
    likelihood_log_pdf_model_1,
    likelihood_log_pdf_model_2,
]

property_subsets = [
    [0], [1], [2],
    [0, 1], [0, 2], [1, 2],
    [0, 1, 2],
]

import numpy as np
from mcmc import random_walk_mh

from scipy.special import logsumexp


def ESS(log_weights):
    """
    TODO: look also at the function whose expectation we're trying to approximate...

    See "Rethinking the effective sample size" https://arxiv.org/abs/1809.04129
    and references therein for some inspiration in this direction...
    """
    log_Z = logsumexp(log_weights)
    weights = np.exp(log_weights - log_Z)
    return 1 / np.sum(weights ** 2)


def binary_search(binary_f, min_val=0, max_val=1, max_iter=20, initial_guess=None,
                  precision_threshold=0,
                  ):
    """binary_f is False from min_val up to unknown_val,
    then True from unknown_val up to max_val. Find unknown val."""

    intervals = [(min_val, max_val)]
    mid_point = 0.5 * (min_val + max_val)
    if type(initial_guess) != type(None):
        mid_point = initial_guess

    for i in range(max_iter):
        if binary_f(mid_point):
            max_val = mid_point
        else:
            min_val = mid_point
        intervals.append((min_val, max_val))
        mid_point = (min_val + max_val) * 0.5
        if (max_val - min_val) <= precision_threshold:
            break
    return mid_point, intervals


def CESS_SMC(initial_particles, log_prior, log_likelihood, thresh=0.99, resample_thresh=0.5, n_mcmc_steps=5,
             initial_stepsize=0.01, stepsize_adaptation=0.5):
    """Select the next lambda increment based on an estimate of the conditional effective sample size

    Question to think about: how to choose thresh?

    References
    ----------
    Based on description of an adaptive SMC approach that appeared in
    Section 2.4.2. of https://arxiv.org/abs/1612.06468,
    which references Del Moral et al., 2012 and Zhou et al., 2015
    introducing and refining the approach.
    """
    lambdas = [0]
    particle_snapshots = [np.array(initial_particles)]
    n_particles = len(initial_particles)
    log_q_s = [np.array(list(map(log_prior, initial_particles)))]
    current_log_weights = [np.zeros(n_particles)]
    incremental_log_weights = []
    stepsizes = [initial_stepsize]
    acceptance_rates = []

    while lambdas[-1] < 1:

        log_prior_vals = np.array(list(map(log_prior, particle_snapshots[-1])))
        log_lik_vals = np.array(list(map(log_likelihood, particle_snapshots[-1])))

        def get_ESS_at_lambda(lam):
            new_log_q_s = ((1 - lam) * log_prior_vals) + (lam * (log_prior_vals + log_lik_vals))
            log_weights = (new_log_q_s - log_q_s[-1])  # + current_log_weights[-1]
            return ESS(log_weights)

        def too_far(lam):
            return (get_ESS_at_lambda(lam) / n_particles) <= thresh

        # see if we can't jump directly to the end
        if too_far(1.0):
            # if available, guess that the next step will be of the same length as the previous step
            if len(lambdas) < 2:
                initial_guess = 0.5
            else:
                initial_guess = min(1.0, lambdas[-1] + (lambdas[-1] - lambdas[-2]))

            next_lam, intervals = binary_search(too_far,
                                                min_val=lambdas[-1],
                                                max_iter=100,
                                                initial_guess=initial_guess)
        else:
            next_lam = 1.0
        lambdas.append(next_lam)
        print('lambda_{}={}'.format(len(lambdas), lambdas[-1]))
        print('\tlambda_{} - lambda_{} = {}'.format(len(lambdas), len(lambdas) - 1, lambdas[-1] - lambdas[-2]))
        remaining_iterations = int((1.0 - lambdas[-1]) / (lambdas[-1] - lambdas[-2]))
        print('\texpected remaining # iterations: {}'.format(remaining_iterations))

        particles = np.array(particle_snapshots[-1])

        log_q_s_next = ((1 - next_lam) * log_prior_vals) + (next_lam * (log_prior_vals + log_lik_vals))
        incremental_log_weights.append(log_q_s_next - log_q_s[-1])

        log_weights = current_log_weights[-1] + incremental_log_weights[-1]
        if (ESS(log_weights) / len(log_weights)) < resample_thresh:
            log_Z = logsumexp(log_weights)
            weights = np.exp(log_weights - log_Z)

            p = weights / np.sum(weights)
            resample_inds = np.random.choice(np.arange(len(particles)), len(particles), p=p)
            particles = particles[resample_inds]

            log_weights = np.ones(n_particles) * (log_Z - np.log(n_particles))

        current_log_weights.append(log_weights)

        new_particles = []
        acceptance_probs = []
        for i in range(len(particles)):
            log_prob_fun = lambda theta: log_prior(theta) + next_lam * log_likelihood(theta)
            mh_traj, _, accept_prob = random_walk_mh(particles[i], log_prob_fun, n_steps=n_mcmc_steps,
                                                     stepsize=stepsizes[-1],
                                                     progress_bar=False)
            acceptance_probs.append(accept_prob)

            new_particles.append(mh_traj[-1])
        acceptance_rates.append(np.mean(acceptance_probs))

        print('\tstepsize={}, acceptance_rate={}'.format(stepsizes[-1], acceptance_rates[-1]))

        if acceptance_rates[-1] < 0.2:
            new_stepsize = stepsize_adaptation * stepsizes[-1]
        elif acceptance_rates[-1] > 0.8:
            new_stepsize = stepsizes[-1] / stepsize_adaptation
        else:
            new_stepsize = stepsizes[-1]
        stepsizes.append(new_stepsize)

        log_q_s.append(np.array(list(map(log_prob_fun, new_particles))))
        particle_snapshots.append(np.array(new_particles))

    return lambdas, particle_snapshots, log_q_s, incremental_log_weights, current_log_weights, stepsizes[
                                                                                               :-1], acceptance_rates


if __name__ == '__main__':
    np.random.seed(0)
    n_particles = 100
    thresh = 0.99
    n_mcmc_steps = 1
    resample_thresh = 0.9
    for properties in property_subsets:
        for model in models:
            print('running SMC on model {} with property subset {}'.format(model, properties))

            initial_particles = np.array([prior_samplers[model]() for _ in range(n_particles)])

            log_prior = prior_log_pdfs[model]
            log_likelihood = lambda theta: likelihood_log_pdfs[model](theta, properties)

            lambdas, particle_snapshots, log_q_s, incremental_log_weights, current_log_weights, stepsizes, acceptance_rates = \
                CESS_SMC(initial_particles,
                         log_prior=log_prior, log_likelihood=log_likelihood,
                         thresh=thresh, resample_thresh=resample_thresh, n_mcmc_steps=n_mcmc_steps)

            forward_works = - current_log_weights[-1]
            log_Z_estimate = (logsumexp(- forward_works) - np.log(len(forward_works)))
            print('log_Z_estimate: {:.4f}'.format(log_Z_estimate))

            np.savez('smc/cess_smc_model={},property_subset={}.npz'.format(model, properties),
                     lambdas=lambdas,
                     model=model,
                     property_subset=properties,
                     n_particles=n_particles,
                     thresh=thresh,
                     n_mcmc_steps=n_mcmc_steps,
                     resample_thresh=resample_thresh,
                     particle_snapshots=particle_snapshots,
                     log_q_s=np.array(log_q_s),
                     incremental_log_weights=np.array(incremental_log_weights),
                     current_log_weights=np.array(current_log_weights),
                     stepsizes=stepsizes,
                     acceptance_rates=acceptance_rates,
                     )
