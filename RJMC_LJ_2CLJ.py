"""
    This code performs Reversible Jump Markov Chain Monte Carlo for the 
    Lennard-Jones fluid. The target property is heat of vaporization, which
    only depends on epsilon. Therefore, the expected outcome is that RJMC 
    favors the single parameter model (just epsilon) over the two parameter
    model (both epsilon and sigma).
    
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
from scipy.optimize import minimize

# Here we have chosen argon as the test case
compound="ethane"
fname = compound+".yaml"

# Load property values for compound

with open(fname) as yfile:
    yfile = yaml.load(yfile)

eps_lit_LJ = yfile["force_field_params"]["eps_lit_LJ"] #[K]
sig_lit_LJ = yfile["force_field_params"]["sig_lit_LJ"] #[nm]
eps_lit_UA = yfile["force_field_params"]["eps_lit_UA"] #[K]
sig_lit_UA = yfile["force_field_params"]["sig_lit_UA"] #[nm]
Lbond_lit_UA = yfile["force_field_params"]["Lbond_lit_UA"] #[nm]
eps_lit_AUA = yfile["force_field_params"]["eps_lit_AUA"] #[K]
sig_lit_AUA = yfile["force_field_params"]["sig_lit_AUA"] #[nm]
Lbond_lit_AUA = yfile["force_field_params"]["Lbond_lit_AUA"] #[nm]
Tc_RP = yfile["physical_constants"]["T_c"] #[K]
rhoc_RP = yfile["physical_constants"]["rho_c"] #[kg/m3]
M_w = yfile["physical_constants"]["M_w"] #[gm/mol]

# Substantiate LennardJones class
Ethane_LJ = LennardJones(M_w)
Ethane_2CLJ = LennardJones_2C(M_w)

# Epsilon and sigma can be obtained from the critical constants
eps_Tc = Ethane_LJ.calc_eps_Tc(Tc_RP) #[K]
sig_rhoc = Ethane_LJ.calc_sig_rhoc(rhoc_RP) #[nm]

# Create functions that return properties for a given model, eps, sig

def rhol_hat_models(Temp,model,eps,sig):
    
    if model == 0: #Single site LJ

        rhol_hat = Ethane_LJ.rhol_hat_LJ(Temp,eps,sig)
        
    elif model == 1: #Two center UA LJ

        rhol_hat = Ethane_2CLJ.rhol_hat_2CLJQ(Temp,eps,sig,Lbond_lit_UA,0) 

    elif model == 2: #Two center AUA LJ
    
        rhol_hat = Ethane_2CLJ.rhol_hat_2CLJQ(Temp,eps,sig,Lbond_lit_AUA,0) 
    
    return rhol_hat #[kg/m3]       
  
def Psat_hat_models(Temp,model,eps,sig):
    
    if model == 0: #Single site LJ

        Psat_hat = Ethane_LJ.Psat_hat_LJ(Temp,eps,sig)
        
    elif model == 1: #Two center UA LJ

        Psat_hat = Ethane_2CLJ.Psat_hat_2CLJQ(Temp,eps,sig,Lbond_lit_UA,0) 

    elif model == 2: #Two center AUA LJ
    
        Psat_hat = Ethane_2CLJ.Psat_hat_2CLJQ(Temp,eps,sig,Lbond_lit_AUA,0) 
    
    return Psat_hat #[kPa]       

# Load REFPROP data from file so that user does not need REFPROP
data = np.loadtxt('TRC_deltaHv.txt')
T_deltaHv = data[:,0] #[K]
RP_deltaHv = data[:,1] #[kJ/mol]

data = np.loadtxt('TRC_data_rhoL.txt')
T_rhol_data = data[:,0] #[K]
rhol_data = data[:,1] #[kJ/mol]

data = np.loadtxt('TRC_data_Pv.txt')
T_Psat_data = data[:,0] #[K]
Psat_data = data[:,1] #[kJ/mol]

# Limit temperature range to that which is typical of ITIC MD simulations

T_min = 137
T_max = 260

rhol_data = rhol_data[T_rhol_data>T_min]
T_rhol_data = T_rhol_data[T_rhol_data>T_min]
rhol_data = rhol_data[T_rhol_data<T_max]
T_rhol_data = T_rhol_data[T_rhol_data<T_max]

Psat_data = Psat_data[T_Psat_data>T_min]
T_Psat_data = T_Psat_data[T_Psat_data>T_min]
Psat_data = Psat_data[T_Psat_data<T_max]
T_Psat_data = T_Psat_data[T_Psat_data<T_max]

# Set percent uncertainty in each property
# These values are to represent the simulation uncertainty more than the experimental uncertainty
# Also, the transiton matrix for eps and sig for each model are tuned to this rhol uncertainty.
# I.e. the optimal "lit" values agree well with a 3% uncertainty in rhol. This improved the RJMC model swap acceptance.
pu_rhol = 3
pu_Psat = 5

# I decided to include the same error model I am using for Mie lambda-6
# For pu_rhol_low = 0.3 and pu_rhol_high = 0.5 AUA is 100%
# For pu_rhol_low = 1 and pu_rhol_high = 3 LJ 16%, UA 22%, AUA 62%
#pu_rhol_low = 1
#T_rhol_switch = 230
#pu_rhol_high = 3
#
#pu_Psat_low = 5
#T_Psat_switch = 180
#pu_Psat_high = 3
#
## Piecewise function to represent the uncertainty in rhol and Psat
#pu_rhol = np.piecewise(T_rhol_data,[T_rhol_data<T_rhol_switch,T_rhol_data>=T_rhol_switch],[pu_rhol_low,lambda x:np.poly1d(np.polyfit([T_rhol_switch,T_max],[pu_rhol_low,pu_rhol_high],1))(x)])
#pu_Psat = np.piecewise(T_Psat_data,[T_Psat_data<T_Psat_switch,T_Psat_data>=T_Psat_switch],[lambda x:np.poly1d(np.polyfit([T_min,T_Psat_switch],[pu_Psat_low,pu_Psat_high],1))(x),pu_Psat_high])
  
# Calculate the absolute uncertainty
u_rhol = rhol_data*pu_rhol/100.
u_Psat = Psat_data*pu_Psat/100.
   
# Calculate the estimated standard deviation
sd_rhol = u_rhol/2.
sd_Psat = u_Psat/2.

# Calculate the precision in each property
t_rhol = np.sqrt(1./sd_rhol)
t_Psat = np.sqrt(1./sd_Psat)                                   

# Initial values for the Markov Chain
guess_0 = (0,eps_lit_LJ, sig_lit_LJ) # Can use critical constants
guess_1 = (1,eps_lit_UA,sig_lit_UA)
guess_2 = (2,eps_lit_AUA,sig_lit_AUA)

## These transition matrices are designed for when rhol is the only target property
#
#Tmatrix_eps = np.ones([3,3])
#Tmatrix_eps[0,1] = eps_lit_UA/eps_lit_LJ
#Tmatrix_eps[0,2] = eps_lit_AUA/eps_lit_LJ
#Tmatrix_eps[1,0] = eps_lit_LJ/eps_lit_UA
#Tmatrix_eps[1,2] = eps_lit_AUA/eps_lit_UA
#Tmatrix_eps[2,0] = eps_lit_LJ/eps_lit_AUA
#Tmatrix_eps[2,1] = eps_lit_UA/eps_lit_AUA
#           
#Tmatrix_sig = np.ones([3,3])
#Tmatrix_sig[0,1] = sig_lit_UA/sig_lit_LJ
#Tmatrix_sig[0,2] = sig_lit_AUA/sig_lit_LJ
#Tmatrix_sig[1,0] = sig_lit_LJ/sig_lit_UA
#Tmatrix_sig[1,2] = sig_lit_AUA/sig_lit_UA
#Tmatrix_sig[2,0] = sig_lit_LJ/sig_lit_AUA
#Tmatrix_sig[2,1] = sig_lit_UA/sig_lit_AUA           
           
# Initial estimates for standard deviation used in proposed distributions of MCMC
guess_var = [1,20, 0.05]
# Variance (or standard deviation, need to verify which one it is) in priors for epsilon and sigma
#prior_var = [5,0.001]

# Simplify notation
dnorm = distributions.norm.logpdf
dgamma = distributions.gamma.logpdf
duni = distributions.uniform.logpdf

rnorm = np.random.normal
runif = np.random.rand

properties = 'rhol'

def calc_posterior(model,eps, sig):

    logp = 0
#    print(eps,sig)
    # Using noninformative priors
    logp += duni(sig, 0, 1)
    logp += duni(eps, 0,1000) 
    
#    print(eps,sig)
    rhol_hat = rhol_hat_models(T_rhol_data,model,eps,sig) #[kg/m3]
    Psat_hat = Psat_hat_models(T_Psat_data,model,eps,sig) #[kPa]        
 
    # Data likelihood
    if properties == 'rhol':
        logp += sum(dnorm(rhol_data,rhol_hat,t_rhol**-2.))
    elif properties == 'Psat':
        logp += sum(dnorm(Psat_data,Psat_hat,t_Psat**-2.))
    elif properties == 'Multi':
        logp += sum(dnorm(rhol_data,rhol_hat,t_rhol**-2.))
        logp += sum(dnorm(Psat_data,Psat_hat,t_Psat**-2.))
    return logp

def gen_Tmatrix():
    ''' Generate Transition matrices based on the optimal eps, sig for different models'''
    
    obj_LJ = lambda eps_sig: -calc_posterior(0,eps_sig[0],eps_sig[1])
    obj_UA = lambda eps_sig: -calc_posterior(1,eps_sig[0],eps_sig[1])
    obj_AUA = lambda eps_sig: -calc_posterior(2,eps_sig[0],eps_sig[1])
    
    guess_LJ = [guess_0[1],guess_0[2]]
    guess_UA = [guess_1[1],guess_1[2]]
    guess_AUA = [guess_2[1],guess_2[2]]
    
    # Make sure bounds are in a reasonable range so that models behave properly
    bnd_LJ = ((0.95*guess_0[1],guess_0[1]*1.05),(0.99*guess_0[2],guess_0[2]*1.01))
    bnd_UA = ((0.95*guess_1[1],guess_1[1]*1.05),(0.99*guess_1[2],guess_1[2]*1.01))
    bnd_AUA = ((0.95*guess_2[1],guess_2[1]*1.05),(0.99*guess_2[2],guess_2[2]*1.01))
    
    #Help debug
#    print(bnd_LJ)
#    print(bnd_UA)
#    print(bnd_AUA)
    
    opt_LJ = minimize(obj_LJ,guess_LJ,bounds=bnd_LJ)
    opt_UA = minimize(obj_UA,guess_UA,bounds=bnd_UA)
    opt_AUA = minimize(obj_AUA,guess_AUA,bounds=bnd_AUA)
    
    #Help debug
#    print(opt_LJ)
#    print(opt_UA)
#    print(opt_AUA)
        
    eps_opt_LJ, sig_opt_LJ = opt_LJ.x[0], opt_LJ.x[1]
    eps_opt_UA, sig_opt_UA = opt_UA.x[0], opt_UA.x[1]
    eps_opt_AUA, sig_opt_AUA = opt_AUA.x[0], opt_AUA.x[1]

    Tmatrix_eps = np.ones([3,3])
    Tmatrix_eps[0,1] = eps_opt_UA/eps_opt_LJ
    Tmatrix_eps[0,2] = eps_opt_AUA/eps_opt_LJ
    Tmatrix_eps[1,0] = eps_opt_LJ/eps_opt_UA
    Tmatrix_eps[1,2] = eps_opt_AUA/eps_opt_UA
    Tmatrix_eps[2,0] = eps_opt_LJ/eps_opt_AUA
    Tmatrix_eps[2,1] = eps_opt_UA/eps_opt_AUA
               
    Tmatrix_sig = np.ones([3,3])
    Tmatrix_sig[0,1] = sig_opt_UA/sig_opt_LJ
    Tmatrix_sig[0,2] = sig_opt_AUA/sig_opt_LJ
    Tmatrix_sig[1,0] = sig_opt_LJ/sig_opt_UA
    Tmatrix_sig[1,2] = sig_opt_AUA/sig_opt_UA
    Tmatrix_sig[2,0] = sig_opt_LJ/sig_opt_AUA
    Tmatrix_sig[2,1] = sig_opt_UA/sig_opt_AUA 
               
    return Tmatrix_eps, Tmatrix_sig

Tmatrix_eps, Tmatrix_sig = gen_Tmatrix()

def RJMC_tuned(calc_posterior,n_iterations, initial_values, prop_var, 
                     tune_for=None, tune_interval=100):
    
    n_params = len(initial_values) #One column is the model number
            
    # Initial proposal standard deviations
    prop_sd = prop_var
    
    # Initialize trace for parameters
    trace = np.zeros((n_iterations+1, n_params)) #n_iterations + 1 to account for guess
    logp_trace = np.zeros(n_iterations+1)
    # Set initial values
    trace[0] = initial_values

    # Initialize acceptance counts
    accepted = [0]*n_params
    rejected = [0]*n_params
               
    model_swaps = 0
    model_swap_attempts = 0
    swap_freq = 1
    
    # Calculate joint posterior for initial values
    current_log_prob = calc_posterior(*trace[0])
    
    logp_trace[0] = current_log_prob
    
    if tune_for is None:
        tune_for = n_iterations/2
    
    for i in range(n_iterations):
    
        if not i%1000: print('Iteration '+str(i))
    
        # Grab current parameter values
        current_params = trace[i].copy()
        trace[i+1] = current_params.copy() #Initialize the next step with the current step. Then update if MCMC move is accepted
        current_model = int(current_params[0])
        logp_trace[i+1] = current_log_prob.copy()
        
        # Loop through model parameters
        
        for j in range(n_params):
    
            # Get current value for parameter j
            params = current_params.copy() # This approach updates previous param values
            
            # Propose new values
            if j == 0: #If proposing a new model
                if not i%swap_freq:
                    mod_ran = np.random.random()
                    if mod_ran < 1./3: #Use new models with equal probability
                        proposed_model = 0
                    elif mod_ran < 2./3:
                        proposed_model = 1
                    elif mod_ran < 1:
                        proposed_model = 2
                    if proposed_model != current_model:
                        model_swap_attempts += 1
                        params[0] = proposed_model
                        params[1] *= Tmatrix_eps[current_model,proposed_model]
                        params[2] *= Tmatrix_sig[current_model,proposed_model]
            else:        
                params[j] = rnorm(current_params[j], prop_sd[j])
    
            # Calculate log posterior with proposed value
            proposed_log_prob = calc_posterior(*params)
    
            # Log-acceptance rate (all other terms in RJMC are 1 in this case)
            alpha = proposed_log_prob - current_log_prob
 
            # Sample a uniform random variate (urv)
            urv = runif()
    
            # Test proposed value
            if np.log(urv) < alpha:
                
                # Accept
                trace[i+1] = params
                logp_trace[i+1] = proposed_log_prob.copy()
                current_log_prob = proposed_log_prob.copy()
                current_params = params
                accepted[j] += 1
                if j == 0:
                    if proposed_model != current_model:
                        model_swaps += 1
                        
            else:
                # Reject
                rejected[j] += 1
            
            # Tune every 100 iterations
            if (not (i+1) % tune_interval) and (i < tune_for) and j != 0:

                acceptance_rate = (1.*accepted[j])/tune_interval             
                if acceptance_rate<0.2:
                    prop_sd[j] *= 0.9
                elif acceptance_rate>0.5:
                    prop_sd[j] *= 1.1                  

                accepted[j] = 0              

    accept_prod = np.array(accepted)/(np.array(accepted)+np.array(rejected))                    

    print('Proposed standard deviations are: '+str(prop_sd))
                
    return trace, trace[tune_for:], logp_trace, logp_trace[tune_for:],accept_prod, model_swaps

# Set the number of iterations to run RJMC and how long to tune for
n_iter = 20000 # 20000 appears to be sufficient
tune_for = 10000 #10000 appears to be sufficient
trace_all,trace_tuned,logp_all,logp_tuned, acc_tuned, model_swaps = RJMC_tuned(calc_posterior, n_iter, guess_0, prop_var=guess_var, tune_for=tune_for)

model_params = trace_all[tune_for+1:,0]

# Converts the array with number of model parameters into an array with the number of times there was 1 parameter or 2 parameters
model_count = np.array([len(model_params[model_params==0]),len(model_params[model_params==1]),len(model_params[model_params==2])])

print('Acceptance Rate during production for eps, sig: '+str(acc_tuned[1:]))

print('Acceptance model swap during production: '+str(model_swaps/(n_iter-tune_for)))

prob_0 = 1.*model_count[0]/(n_iter-tune_for)
print('Percent that single site LJ model is sampled: '+str(prob_0 * 100.)) #The percent that use 1 parameter model

prob_1 = 1.*model_count[1]/(n_iter-tune_for)
print('Percent that two-center UA LJ model is sampled: '+str(prob_1 * 100.)) #The percent that use 1 parameter model
     
prob_2 = 1.*model_count[2]/(n_iter-tune_for)
print('Percent that two-center AUA LJ model is sampled: '+str(prob_2 * 100.)) #The percent that use 1 parameter model
     
# Create plots of the Markov Chain values for epsilon, sigma, and precision     
f, axes = plt.subplots(3, 2, figsize=(10,10))     
for param, samples, samples_tuned, iparam in zip(['model','$\epsilon (K)$', '$\sigma (nm)$'], trace_all.T,trace_tuned.T, [0,1,2]):
    axes[iparam,0].plot(samples)
    axes[iparam,0].set_ylabel(param)
    axes[iparam,0].set_xlabel('Iteration')
    axes[iparam,1].hist(samples_tuned)
    axes[iparam,1].set_xlabel(param)
    axes[iparam,1].set_ylabel('Count')
    
plt.tight_layout(pad=0.2)

f.savefig(compound+"_Trace_RJMC.pdf")

# Plot logp
f = plt.figure()
plt.semilogy(-logp_all)
plt.xlabel('Iteration')
plt.ylabel('-logPosterior')

f.savefig(compound+"_logp_RJMC.pdf")  


# Plot the eps and sig parameters that are sampled and compare with literature, critical point, and guess values
f = plt.figure()
plt.scatter(trace_all[:,2],trace_all[:,1],label='Trajectory')
plt.scatter(trace_tuned[:,2],trace_tuned[:,1],label='Production')
#plt.scatter(sig_lit,eps_lit,label='Literature')
#plt.scatter(sig_rhoc,eps_Tc,label='Critical Point')
plt.scatter(guess_0[2],guess_0[1],label='Guess LJ')
plt.scatter(guess_1[2],guess_1[1],label='Guess 2CLJ UA')
plt.scatter(guess_2[2],guess_2[1],label='Guess 2CLJ AUA')
plt.xlabel('$\sigma (nm)$')
plt.ylabel('$\epsilon (K)$')
plt.legend()

f.savefig(compound+"_Trajectory_RJMC.pdf")  

T_plot_deltaHv = np.linspace(T_deltaHv.min(), T_deltaHv.max())
T_plot_rhol = np.linspace(T_rhol_data.min(), T_rhol_data.max())
T_plot_Psat = np.linspace(T_Psat_data.min(), T_Psat_data.max())
   
# Plot the predicted properties versus REFPROP. Include the Bayesian uncertainty by sampling a subset of 100 eps/sig.
f, axarr = plt.subplots(2,2,figsize=(10,10))

color_scheme = ['b','r','g']
labels = ['LJ','2CLJ UA','2CLJ AUA']
xlabels = ['$T$ (K)','$T$ (K)','$1000/T$ (K)','$T$ (K)']

for i in range(100): #Plot 100 random samples from production
    model_sample, eps_sample, sig_sample = trace_tuned[np.random.randint(0, n_iter - tune_for)]
    model_sample = int(model_sample)
    if model_sample == 0:
        deltaHv_sample = Ethane_LJ.deltaHv_hat_LJ(T_plot_deltaHv,eps_sample)
        axarr[1,1].plot(T_plot_deltaHv,deltaHv_sample,color_scheme[model_sample])
    rhol_sample = rhol_hat_models(T_plot_rhol,model_sample,eps_sample,sig_sample)
    Psat_sample = Psat_hat_models(T_plot_Psat,model_sample,eps_sample,sig_sample)

    axarr[0,0].plot(T_plot_rhol,rhol_sample,color_scheme[model_sample],alpha=0.5)
    axarr[1,0].plot(T_plot_Psat,np.log10(Psat_sample),color_scheme[model_sample],alpha=0.5)
    axarr[0,1].plot(1000./T_plot_Psat,np.log10(Psat_sample),color_scheme[model_sample],alpha=0.5) 

axarr[0,0].plot(T_rhol_data,rhol_data,'ko',mfc='None',label='TRC')
axarr[1,0].plot(T_Psat_data,np.log10(Psat_data),'ko',mfc='None',label='TRC')
axarr[0,1].plot(1000./T_Psat_data,np.log10(Psat_data),'ko',mfc='None',label='TRC')
axarr[1,1].plot(T_deltaHv,RP_deltaHv,'k--',label='REFPROP')

for axrow in axarr:
    for ax in axrow:
        for col, lab in zip(color_scheme,labels):
            ax.plot([],[],col,label=lab)
        ax.legend()
    
axarr[0,0].set_xlabel("$T$ (K)")
axarr[0,0].set_ylabel(r"$\rho_l \left(\frac{kg}{m3}\right)$")

axarr[0,0].set_xlabel("$T$ (K)")
axarr[1,0].set_ylabel(r"$log_{10}\left(\frac{P_{sat}}{kPa}\right)$")

axarr[0,0].set_xlabel("$1000/T$ (K)")
axarr[0,1].set_ylabel(r"$log_{10}\left(\frac{P_{sat}}{kPa}\right)$")

axarr[1,1].set_xlabel("$T$ (K)")
axarr[1,1].set_ylabel(r"$\Delta H_v \left(\frac{kJ}{mol}\right)$")

plt.tight_layout(pad=0.2)

f.savefig(compound+"_Prop_RJMC.pdf")
 