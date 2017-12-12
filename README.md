# RJMC_LJ_Ethane

Reversible Jump MCMC for the Ethane. Three models are compared, single-site Lennard-Jones, two-center united-atom Lennard-Jones, two-center anisotropic-united-atom Lennard-Jones

The target property is saturated liquid density between 137-260K

The main code is "RJMC_LJ_2CLJ.py". This file:

1)Extracts compound specific values for ethane from "ethane.yaml"

2)Reads in the experimental data from TRC_data_rhol.txt and TRC_data_Pv.txt

3) Imports a class for the single-site LJ fluid from "LennardJones_correlations.py". Import a class for the two-center LJ fluid from "LennardJones_2Center_correlations.py". These classes provide functions that can predict properties such as liquid density and vapor pressure. The necessary correlation parameters are found in "LJ_fluid.yaml" and "DCLJQ_fluid.yaml".

4) Contains two primary modules: a) Calc_posterior which calculates the posterior distribution for a given epsilon and sigma b) RJMC_tuned which changes the values of epsilon and sigma according to a random walk Markov Chain

Each model has been given an equal probability. The Jacobian needs to be determined.

The results vary more than they probably should. But a typical succesful run should look like:

Acceptance Rate during production for eps, sig: [ 0.21968739  0.17964607]

Acceptance model swap during production: 0.3545

Percent that single site LJ model is sampled: 30.21

Percent that two-center UA LJ model is sampled: 30.85

Percent that two-center AUA LJ model is sampled: 38.94

Typical plots of the results are found as PDF files.

ethane_Trace_RJMC.pdf depicts the trace for model, epsilon, sigma along with histograms. Note that model = 0 is single-site LJ, model = 1 is two-center UA LJ, model = 2 is two-center AUA LJ. Histograms are not very informative as currently they plot each model simultaneously, which leads to very large bin sizes.

ethane_Prop_RJMC.pdf presents the feasible regions for liquid density, vapor pressure, and (in the case of single-site LJ) heat of vaporization. Over this small temperature region each model should represent liquid density accurately.

ethane_Trajectory_RJMC.pdf depicts the trajectory through the parameter space of accepted moves. Three different regions of epsilon and sigma correspond to the three models. 

The primary questions are:


Is our acceptance criterion rigorous? Specifically, how do we include the Jacobian term? Currently, I use a ratio to convert epsilon and sigma of one model to a different model. The ratio is based on the optimal epsilon and sigma for each model. Does this ratio need to be included somewhere in the acceptance criterion?

How can we improve the algorithm? For example, the frequency of model swaps, the transition matrix.

Do we need to have different proposed standard deviations in epsilon and sigma for the different models?

Documentation:

The documentation for the single-site LJ model equations is found in LJ_fluid_correlation_Lotfi.pdf.

The documentation for the two-center LJ model is found in 2CLJQ_fluid_correlations_Stoll.pdf.