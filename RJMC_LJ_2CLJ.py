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
from scipy.stats import linregress
from scipy.optimize import minimize
import random as rm

# Here we have chosen ethane as the test case
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
eps_lit2_AUA = yfile["force_field_params"]["eps_lit2_AUA"] #[K]
sig_lit2_AUA = yfile["force_field_params"]["sig_lit2_AUA"] #[nm]
Lbond_lit2_AUA = yfile["force_field_params"]["Lbond_lit2_AUA"] #[nm]
Q_lit2_AUA = yfile["force_field_params"]["Q_lit2_AUA"] #[DAng]
eps_lit3_AUA = yfile["force_field_params"]["eps_lit3_AUA"] #[K]
sig_lit3_AUA = yfile["force_field_params"]["sig_lit3_AUA"] #[nm]
Lbond_lit3_AUA = yfile["force_field_params"]["Lbond_lit3_AUA"] #[nm]
Q_lit3_AUA = yfile["force_field_params"]["Q_lit3_AUA"] #[DAng]
Tc_RP = yfile["physical_constants"]["T_c"] #[K]
rhoc_RP = yfile["physical_constants"]["rho_c"] #[kg/m3]
M_w = yfile["physical_constants"]["M_w"] #[gm/mol]

#%%

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
        
    elif model == 3: #Two center AUA LJ+Q
    
        rhol_hat = Ethane_2CLJ.rhol_hat_2CLJQ(Temp,eps,sig,Lbond_lit2_AUA,Q_lit2_AUA) 
        
    elif model == 4: #Two center AUA LJ+Q=0
    
        rhol_hat = Ethane_2CLJ.rhol_hat_2CLJQ(Temp,eps,sig,Lbond_lit3_AUA,Q_lit3_AUA) 
        
    elif model == 5: #Two center AUA LJ with L correlation
    
        sigma0 = Lbond_lit_AUA+2.68*sig_lit_AUA
        Lfit = sigma0 - 2.68*sig
    
        rhol_hat = Ethane_2CLJ.rhol_hat_2CLJQ(Temp,eps,sig,Lfit,0)
    
    return rhol_hat #[kg/m3]       
  
def Psat_hat_models(Temp,model,eps,sig):
    
    if model == 0: #Single site LJ

        Psat_hat = Ethane_LJ.Psat_hat_LJ(Temp,eps,sig)
        
    elif model == 1: #Two center UA LJ

        Psat_hat = Ethane_2CLJ.Psat_hat_2CLJQ(Temp,eps,sig,Lbond_lit_UA,0) 

    elif model == 2: #Two center AUA LJ
    
        Psat_hat = Ethane_2CLJ.Psat_hat_2CLJQ(Temp,eps,sig,Lbond_lit_AUA,0) 
        
    elif model == 3: #Two center AUA LJ+Q
    
        Psat_hat = Ethane_2CLJ.Psat_hat_2CLJQ(Temp,eps,sig,Lbond_lit2_AUA,Q_lit2_AUA) 
        
    elif model == 4: #Two center AUA LJ+Q=0
    
        Psat_hat = Ethane_2CLJ.Psat_hat_2CLJQ(Temp,eps,sig,Lbond_lit3_AUA,Q_lit3_AUA) 
        
    elif model == 5: #Two center AUA LJ with L correlation
    
        sigma0 = Lbond_lit_AUA+2.68*sig_lit_AUA
        Lfit = sigma0 - 2.68*sig
    
        Psat_hat = Ethane_2CLJ.Psat_hat_2CLJQ(Temp,eps,sig,Lfit,0) 
    
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
pu_Psat = 20

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
guess_3 = (3,eps_lit2_AUA,sig_lit2_AUA)
guess_4 = (4,eps_lit3_AUA,sig_lit3_AUA)
guess_5 = (5,eps_lit_AUA,sig_lit_AUA)

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


#OCM: All of this first section is Rich's data setup, which I don't have any reason to alter.  I am focusing more on the monte carlo implementation
#%%

T_lin=np.linspace(T_min,T_max,num=186)

rhol_fake_data_model0=rhol_hat_models(T_lin,0,eps_lit_LJ, sig_lit_LJ)
rhol_fake_data_model1=rhol_hat_models(T_lin,1,eps_lit_UA,sig_lit_UA)
rhol_fake_data_model2=rhol_hat_models(T_lin,2,eps_lit_AUA,sig_lit_AUA)
#plt.plot(T_rhol_data,rhol_fake_data_model0)
#plt.plot(T_rhol_data,rhol_fake_data_model1)
#plt.plot(T_rhol_data,rhol_fake_data_model2)
rhol_fake_data_mix=np.empty(np.size(T_lin))
num0=0
num1=0
for i in range(np.size(T_rhol_data)):
    randi=np.random.random()
    if randi <= 0.3:
        rhol_fake_data_mix[i]=rhol_fake_data_model0[i]
        num0+=1
    elif 0.3 < randi <= 0.6:
        rhol_fake_data_mix[i]=rhol_fake_data_model1[i]
        num1+=1
    else:
        rhol_fake_data_mix[i]=rhol_fake_data_model2[i]
print(num0)
print(num1)

#OCM: This was me attempting to create a fake dataset so that I could reliably reproduce sampling with the ratios of data that I made
#%%
# Simplify notation
dnorm = distributions.norm.logpdf
dgamma = distributions.gamma.logpdf
duni = distributions.uniform.logpdf

rnorm = np.random.normal
runif = np.random.rand

properties = 'Psat'

def calc_posterior(model,eps, sig):

    logp = 0
#    print(eps,sig)
    # Using noninformative priors
    logp += duni(sig, 0, 1)
    logp += duni(eps, 0,500) 
    # OCM: no reason to use anything but uniform priors at this point.  Could probably narrow the prior ranges a little bit to improve acceptance,
    #But Rich is rightly being conservative here especially since evaluations are cheap.
    
#    print(eps,sig)
    #rhol_hat_fake = rhol_hat_models(T_lin,model,eps,sig)
    rhol_hat = rhol_hat_models(T_rhol_data,model,eps,sig) #[kg/m3]
    Psat_hat = Psat_hat_models(T_Psat_data,model,eps,sig) #[kPa]        
 
    # Data likelihood
    if properties == 'rhol':
        logp += sum(dnorm(rhol_data,rhol_hat,t_rhol**-2.))
        #logp += sum(dnorm(rhol_data,rhol_hat,t_rhol**-2.))
    elif properties == 'Psat':
        logp += sum(dnorm(Psat_data,Psat_hat,t_Psat**-2.))
    elif properties == 'Multi':
        logp += sum(dnorm(rhol_data,rhol_hat,t_rhol**-2.))
        logp += sum(dnorm(Psat_data,Psat_hat,t_Psat**-2.))
    return logp
    #return rhol_hat
    
    #OCM: Standard calculation of the log posterior. Note that t_rhol and t_Psat are precisions
    #This is one of the most important areas of the code for testing as it is where you can substitute in different data for training or change what property it is training on.

def gen_Tmatrix():
    ''' Generate Transition matrices based on the optimal eps, sig for different models'''
    
    obj_LJ = lambda eps_sig: -calc_posterior(0,eps_sig[0],eps_sig[1])
    obj_UA = lambda eps_sig: -calc_posterior(1,eps_sig[0],eps_sig[1])
    obj_AUA = lambda eps_sig: -calc_posterior(2,eps_sig[0],eps_sig[1])
    
    guess_LJ = [guess_0[1],guess_0[2]]
    guess_UA = [guess_1[1],guess_1[2]]
    guess_AUA = [guess_2[1],guess_2[2]]
    
    # Make sure bounds are in a reasonable range so that models behave properly
    bnd_LJ = ((0.75*guess_0[1],guess_0[1]*1.25),(0.90*guess_0[2],guess_0[2]*1.1))
    bnd_UA = ((0.75*guess_1[1],guess_1[1]*1.25),(0.90*guess_1[2],guess_1[2]*1.1))
    bnd_AUA = ((0.75*guess_2[1],guess_2[1]*1.25),(0.90*guess_2[2],guess_2[2]*1.1))
    
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
        
    #OCM: Important distinction:  This is not the transition matrix in the tradition RJMC sense of the term, which may be confusing.
    #This is a map between the 3 different probability spaces/models.  We will use this to change our variables, and as part of the jacobian determinant to change the acceptance ratio.
    #This RJMC problem of disjoint high probability regions will likely be exacerbated with a high dimensional and variable model space.
    #Will probably require a more sophisticated solution like AIS-RJMC (https://www.tandfonline.com/doi/abs/10.1080/10618600.2013.805651)
    #But I expect that the approach here will work for relatively simple problems such as this one

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
               
    return Tmatrix_eps, Tmatrix_sig, eps_opt_LJ,eps_opt_UA,eps_opt_AUA,sig_opt_LJ,sig_opt_UA,sig_opt_AUA

Tmatrix_eps, Tmatrix_sig, eps_opt_LJ,eps_opt_UA,eps_opt_AUA,sig_opt_LJ,sig_opt_UA,sig_opt_AUA = gen_Tmatrix()

### Performs a 2-dimensional scan and integration of parameter space

twoD_scan = True
neps = 100
nsig = 100
scan_method = 1 #Method = 0, all scans use same region, Method = 1, scans use zoomed in region (should be more accurate)

model_list = [0,1,2]

eps_low = {0:230,1:95,2:130}
eps_high = {0:245,1:105,2:140}
sig_low = {0:0.4175,1:0.374,2:0.349}
sig_high = {0:0.422,1:0.38,2:0.355}

logp_scan = {}
logp_sum = {}
logp_opt = {}

for model in model_list: 
    logp_sum[model] = 0
    logp_opt[model] = -1e10

if twoD_scan:

    if scan_method == 0:
        
        eps_scan = np.linspace(90,250,neps)
        sig_scan = np.linspace(0.35,0.43,nsig)
        
        for model in model_list: logp_scan[model] = np.zeros([neps,nsig])
    
        for ieps, eps in enumerate(eps_scan):
            
            for isig, sig in enumerate(sig_scan):
        
                for model in model_list:
                
                    logp = calc_posterior(model,eps,sig)
                    
                    logp_scan[model][ieps,isig] = logp
                    
                    logp_sum[model] += logp
                            
                    if logp > logp_opt[model]:
                            
                        logp_opt[model] = logp

    elif scan_method == 1:
        
        logp_scan = {0:[],1:[],2:[]}
        eps_scan = {0:[],1:[],2:[]}
        sig_scan = {0:[],1:[],2:[]}
        
        deps = 0.05
        dsig = 0.0005
       
        for model in model_list:
            
            neps = int((eps_high[model] - eps_low[model])/deps)
            nsig = int((sig_high[model] - sig_low[model])/dsig)
            
            eps_scan[model] = np.linspace(eps_low[model],eps_high[model],neps)
            sig_scan[model] = np.linspace(sig_low[model],sig_high[model],nsig)
            
            logp_scan[model] = np.zeros([neps,nsig])

            for ieps, eps in enumerate(eps_scan[model]):
                
                for isig, sig in enumerate(sig_scan[model]):
                    
                    logp = calc_posterior(model,eps,sig)
                        
                    logp_scan[model][ieps,isig] = logp
                    
                    logp_sum[model] += logp
                            
                    if logp > logp_opt[model]:
                            
                        logp_opt[model] = logp

### Combine the probabilities relative to the overall optimal
prob_model = {0:[],1:[],2:[]}
prob_model_total = {0:[],1:[],2:[]}
prob_normal_model = 0

for model in model_list:
    
    prob_model[model] = np.exp(logp_scan[model] - logp_opt[model])

    prob_normal_model += np.nansum(prob_model[model])

for model in model_list:
    
    prob_model[model] /= prob_normal_model
              
    prob_model_total[model] = np.nansum(prob_model[model])

print('The relative probabilities of each model are: ')
print(prob_model_total)

if scan_method == 0:

    for model in model_list:
        plt.contour(sig_scan,eps_scan,prob_model[model])
        plt.show()
    
elif scan_method == 1:

    for model in model_list:
        CS = plt.contour(sig_scan[model],eps_scan[model],prob_model[model])
        plt.clabel(CS, inline=1, fontsize=10)
        plt.colorbar(CS)
        plt.xlabel(r'$\sigma$ (nm)')
        plt.ylabel(r'$\epsilon$ (K)')
        plt.show()
        
#%%
def RJMC_outerloop(calc_posterior,n_iterations,initial_values,initial_sd,n_models,swap_freq,tune_freq,tune_for,T_matrix_1,T_matrix_2):
    
    
    #INITIAL SETUP FOR MC LOOP
    #-----------------------------------------------------------------------------------------#
    
    n_params = len(initial_values) #One column is the model number
    accept_vector=np.zeros((n_iterations,3))
    attempt_vector=np.zeros((n_iterations,3))
    prop_sd=initial_sd
    
    #Initialize matrices to count number of moves of each type
    attempt_matrix=np.zeros((n_params,n_models,n_models))
    acceptance_matrix=np.zeros((n_params,n_models,n_models))
    
    
    # Initialize trace for parameters
    trace = np.zeros((n_iterations+1, n_params)) #n_iterations + 1 to account for guess
    logp_trace = np.zeros(n_iterations+1)
    # Set initial values
    trace[0] = initial_values
    
    # Calculate joint posterior for initial values
    current_log_prob = calc_posterior(*trace[0])
    
    logp_trace[0] = current_log_prob
    current_params=trace[0].copy()
    record_acceptance='False'
    #----------------------------------------------------------------------------------------#
    
    #OUTER MCMC LOOP
    
    for i in range(n_iterations):
        if not i%10000: print('Iteration '+str(i))
        
        
        # Grab current parameter values
        current_params = trace[i].copy()
        current_model = int(current_params[0])
        current_log_prob = logp_trace[i].copy()
        
        if i >= tune_for:
            record_acceptance='True'
        
        new_params, new_log_prob, attempt_matrix,acceptance_matrix,acceptance,proposed_param = RJMC_Moves(current_params,current_model,current_log_prob,n_models,swap_freq,n_params,prop_sd,attempt_matrix,acceptance_matrix,T_matrix_1,T_matrix_2,record_acceptance)
        
        
        attempt_vector[i,proposed_param]+=1
        if acceptance == 'True':
            accept_vector[i,proposed_param]+=1
        logp_trace[i+1] = new_log_prob
        trace[i+1] = new_params
        
        if (not (i+1) % tune_freq) and (i < tune_for):
            prop_sd=proposal_tuning(prop_sd,accept_vector,attempt_vector,n_params)
            
           
    attempt_matrix_final=np.sum(attempt_matrix,0)
    acceptance_matrix_final=np.sum(acceptance_matrix,0)       
    trace_tuned=trace[tune_for:]
    logp_trace_tuned=logp_trace[tune_for:]
    return trace,logp_trace,trace_tuned,logp_trace_tuned,attempt_matrix_final,acceptance_matrix_final,prop_sd,accept_vector,attempt_vector,swap_freq
            

            
    
    
def RJMC_Moves(current_params,current_model,current_log_prob,n_models,swap_freq,n_params,prop_sd,attempt_matrix,acceptance_matrix,T_matrix_1,T_matrix_2,record_acceptance):
    
    params = current_params.copy()# This approach updates previous param values
    #Grab a copy of the current params to work with
    #current_log_prob_copy=copy.deepcopy(current_log_prob)
    
    #Roll a dice to decide what kind of move will be suggested
    mov_ran=np.random.random()
    
    if mov_ran <= swap_freq:
        
        params,rjmc_jacobian,proposed_log_prob,proposed_model=model_proposal(current_model,n_models,params,T_matrix_1,T_matrix_2)
        
        alpha = (proposed_log_prob - current_log_prob) + rjmc_jacobian
        
        acceptance=accept_reject(alpha)
        
        if acceptance =='True':
            new_log_prob=proposed_log_prob
            new_params=params
            if record_acceptance == 'True':
                acceptance_matrix[0,current_model,proposed_model]+=1
                attempt_matrix[0,current_model,proposed_model]+=1
        elif acceptance == 'False':
            new_params=current_params
            new_log_prob=current_log_prob
            if record_acceptance == 'True':
                attempt_matrix[0,current_model,proposed_model]+=1
        proposed_param=0
        '''
        move_type = 'Swap'
    else: 
        move_type = 'Trad'
    
        
    if move_type == 'Swap':
        '''
    else:
        params,proposed_log_prob,proposed_param=parameter_proposal(params,n_params,prop_sd)    
        
        alpha = (proposed_log_prob - current_log_prob)
    
        acceptance=accept_reject(alpha)
                    
    
        if acceptance =='True':
            new_log_prob=proposed_log_prob
            new_params=params
            if record_acceptance == 'True':
                 acceptance_matrix[proposed_param,current_model,current_model]+=1
                 attempt_matrix[proposed_param,current_model,current_model]+=1
        elif acceptance == 'False':
             new_params=current_params
             new_log_prob=current_log_prob
             if record_acceptance == 'True':
                attempt_matrix[proposed_param,current_model,current_model]+=1
    
    
    return new_params,new_log_prob,attempt_matrix,acceptance_matrix,acceptance,proposed_param
            
            
            
def accept_reject(alpha):    
    urv=runif()
    if np.log(urv) < alpha:  
        acceptance='True'
    else: 
        acceptance='False'
    return acceptance
        
def model_proposal(current_model,n_models,params,T_matrix_1,T_matrix_2):
    proposed_model=current_model
    while proposed_model==current_model:
        proposed_model=int(np.floor(np.random.random()*n_models))
            
    params[0] = proposed_model
    params[1] *= T_matrix_1[current_model,proposed_model]
    params[2] *= T_matrix_2[current_model,proposed_model]

    proposed_log_prob=calc_posterior(*params)
    rjmc_jacobian =  np.log(T_matrix_1[current_model,proposed_model]) + np.log(T_matrix_2[current_model,proposed_model])
    return params,rjmc_jacobian,proposed_log_prob, proposed_model
    #Switch models and map parameters to new distributions
        
    
def parameter_proposal(params,n_params,prop_sd):
    proposed_param=int(np.ceil(np.random.random()*(n_params-1)))
    params[proposed_param] = rnorm(params[proposed_param], prop_sd[proposed_param])
    proposed_log_prob=calc_posterior(*params)
    return params, proposed_log_prob,proposed_param


def proposal_tuning(prop_sd,accept_vector,attempt_vector,n_params):
    #print('Tuning on step %1.1i' %i)
    #print(np.sum(accept_vector[i-tune_freq:]))
    acceptance_rate = np.sum(accept_vector,0)/np.sum(attempt_vector,0)        
    #print(acceptance_rate)
    for m in range (n_params):
        if m != 0:
           if acceptance_rate[m]<0.2:
               prop_sd[m] *= 0.9
               #print('Yes')
           elif acceptance_rate[m]>0.5:
               prop_sd[m] *= 1.1
               #print('No')
                
    return prop_sd
        
initial_values=(0,eps_lit_LJ, sig_lit_LJ) # Can use critical constants
initial_sd = [1,20, 0.05]
n_iter=50000
tune_freq=100
tune_for=20000
n_models=3
swap_freq=0.2
#The fraction of times a model swap is suggested as the move, rather than an intra-model move
trace,logp_trace,trace_tuned,logp_trace_tuned,attempt_matrix,acceptance_matrix,prop_sd,accept_vector,attempt_vector,swap_freq = RJMC_outerloop(calc_posterior,n_iter,initial_values,initial_sd,n_models,swap_freq,tune_freq,tune_for,Tmatrix_eps,Tmatrix_sig)        
        
#%%        
#print(logp_sum)
#print(logp_opt)


#OCM: With more difficult distributions it might be important to update this distribution as we go along, if it is still viable.



model_params = trace[tune_for+1:,0]

# Converts the array with number of model parameters into an array with the number of times there was 1 parameter or 2 parameters
model_count = np.array([len(model_params[model_params==0]),len(model_params[model_params==1]),len(model_params[model_params==2])])
'''
print('Acceptance Rate during production for eps, sig: '+str(acc_tuned[1:]))

print('Acceptance model swap during production: '+str(model_swaps/(n_iter-tune_for)))
'''
#OCM: Something is wrong with this as it is greater than one, which shouldn't be possible.  Probably just a calculation error that doesn't affect RJMC

prob_0 = 1.*model_count[0]/(n_iter-tune_for)
print('Percent that single site LJ model is sampled: '+str(prob_0 * 100.)) #The percent that use 1 parameter model

prob_1 = 1.*model_count[1]/(n_iter-tune_for)
print('Percent that two-center UA LJ model is sampled: '+str(prob_1 * 100.)) #The percent that use two center UA LJ
     
prob_2 = 1.*model_count[2]/(n_iter-tune_for)
print('Percent that two-center AUA LJ model is sampled: '+str(prob_2 * 100.)) #The percent that use two center AUA LJ

prob_vec=np.asarray([prob_0,prob_1,prob_2])

print('Attempted Moves')
print(attempt_matrix)
print('Accepted Moves')
print(acceptance_matrix)

prob_matrix=acceptance_matrix/attempt_matrix


transition_matrix=np.zeros((n_models,n_models))



for i in range(n_models):
    for j in range(n_models):
        if i != j:
            transition_matrix[i,j]=acceptance_matrix[i,j]/np.sum(attempt_matrix[i,:])
for i in range(n_models):
    transition_matrix[i,i]=1-np.sum(transition_matrix[i,:])
print('Transition Matrix:')
print(transition_matrix)


for i in range(n_models):
    for j in range(n_models):
        if i != j and i < j:
            print('Detailed Balance for model pairing:'+ str(i),str(j))
            print('%7.4f' % (prob_vec[i]*transition_matrix[i,j]))
            print('%7.4f' % (prob_vec[j]*transition_matrix[j,i]))
#%%     
# Create plots of the Markov Chain values for epsilon, sigma, and precision     
f, axes = plt.subplots(3, 2, figsize=(10,10))     
for param, samples, samples_tuned, iparam in zip(['model','$\epsilon (K)$', '$\sigma (nm)$'], trace.T,trace_tuned.T, [0,1,2]):
    axes[iparam,0].plot(samples)
    axes[iparam,0].set_ylabel(param)
    axes[iparam,0].set_xlabel('Iteration')
    axes[iparam,1].hist(samples_tuned,bins=50)
    axes[iparam,1].set_xlabel(param)
    axes[iparam,1].set_ylabel('Count')
    
plt.tight_layout(pad=0.2)

f.savefig(compound+"_Trace_RJMC.pdf")
#%%
trace_0=[]
trace_1=[]
trace_2=[]
for i in range (np.size(trace_tuned,0)):
    if trace_tuned[i,0] == 0:
        trace_0.append(trace_tuned[i])
    if trace_tuned[i,0] == 1:
        trace_1.append(trace_tuned[i])
    if trace_tuned[i,0] == 2:
        trace_2.append(trace_tuned[i])
        
trace_0=np.asarray(trace_0)
trace_1=np.asarray(trace_1)
trace_2=np.asarray(trace_2)
# Plot logp
f = plt.figure()
plt.semilogy(-logp_trace)
plt.xlabel('Iteration')
plt.ylabel('-logPosterior')

f.savefig(compound+"_logp_RJMC.pdf")  
trace_0=np.array(trace_0)
trace_1=np.array(trace_1)
trace_2=np.array(trace_2)

# Plot the eps and sig parameters that are sampled and compare with literature, critical point, and guess values
f = plt.figure(figsize=[5,5])
#plt.scatter(trace_all[:,2],trace_all[:,1],label='Trajectory')
plt.scatter(trace_0[:,2],trace_0[:,1],label='LJ Trajectory',marker='.',color='c')
plt.scatter(trace_1[:,2],trace_1[:,1],label='2CLJ UA Trajectory',marker='.',color='m')
plt.scatter(trace_2[:,2],trace_2[:,1],label='2CLJ AUA Trajectory',marker='.',color='y')
plt.title('RJMC Trajectory, LJ Ethane models')
#plt.scatter(sig_lit,eps_lit,label='Literature')
#plt.scatter(sig_rhoc,eps_Tc,label='Critical Point')
#plt.scatter(guess_0[2],guess_0[1],label='Literature LJ',color='b')
#plt.scatter(guess_1[2],guess_1[1],label='Literature 2CLJ UA',color='r')
#plt.scatter(guess_2[2],guess_2[1],label='Literature 2CLJ AUA',color='g')
plt.xlabel('$\sigma (nm)$',fontsize='large')
plt.ylabel('$\epsilon (K)$',fontsize='large')
plt.legend()

a = plt.axes([0.55, 0.3, 0.3, 0.3])

plt.scatter(trace_0[:,2],trace_0[:,1],marker='.',color='c')
#plt.scatter(guess_0[2],guess_0[1],label='Literature LJ',color='b')
plt.xlim([0.417,0.423])
#a.xlim(220,240)
#a.ylim([0.42,0.425])
a.axes([0.42,0.425,220,240])

plt.xticks([])
plt.yticks([])
plt.show()
f.savefig(compound+"_Trajectory_RJMC.pdf")  
#%%%

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

    axarr[0,0].plot(T_plot_rhol,rhol_sample,color_scheme[model_sample],alpha=0.3)
    axarr[1,0].plot(T_plot_Psat,np.log10(Psat_sample),color_scheme[model_sample],alpha=0.5)
    axarr[0,1].plot(1000./T_plot_Psat,np.log10(Psat_sample),color_scheme[model_sample],alpha=0.5)
    plt.savefig('rjmc_ethane_trajectory',format='pdf')


axarr[0,0].plot(T_lin,rhol_fake_data_model0,ls='--',mfc='None',label='Model 0 Fake Data')
axarr[0,0].plot(T_lin,rhol_fake_data_model1,'--',mfc='None',label='Model 1 Fake Data')
axarr[0,0].plot(T_lin,rhol_fake_data_model2,':',mfc='None',label='Model 2 Fake Data')
#axarr[0,0].plot(T_rhol_data,rhol_data,'ko',mfc='None',label='TRC')
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
plt.show()
f.savefig(compound+"_Prop_RJMC.pdf")
#%%
'''
trace_0=[]
trace_1=[]
trace_2=[]
for i in range(np.size(trace_tuned,0)):
    if trace_tuned[i,0]==0:
        trace_0.append(trace_tuned[i])
    elif trace_tuned[i,0]==1:
        trace_1.append(trace_tuned[i])
    elif trace_tuned[i,0]==2:
        trace_2.append(trace_tuned[i])
#plt.plot(trace_2[:,1],trace_2[:,2])
'''       
'''
trace_0=np.asarray(trace_0)
trace_1=np.asarray(trace_1)
trace_2=np.asarray(trace_2)
if len(trace_0) > 0: plt.scatter(trace_0[:,1],trace_0[:,2],label='LJ')
if len(trace_1) > 0: plt.scatter(trace_1[:,1],trace_1[:,2],label='UA')
if len(trace_2) > 0: plt.scatter(trace_2[:,1],trace_2[:,2],label='AUA')
plt.legend()
'''
#%%
'''
plt.plot(T_lin,rhol_fake_data_model0,ls='--',mfc='None',label='Model 0 Fake Data')
plt.plot(T_lin,rhol_fake_data_model1,'--',mfc='None',label='Model 1 Fake Data')
plt.plot(T_lin,rhol_fake_data_model2,':',mfc='None',label='Model 2 Fake Data')
plt.legend()
plt.show()
'''