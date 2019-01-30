#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 12:15:26 2018

Pseudocode examples for property calculator API


@author: owenmadin
"""

from openforcefield.propertycalculator import protocols
from openforcefield.propertycalculator.protocols import ExtractableStatistics
from openforcefield.properties import PropertyType

#Use case 1: Bayesian inference over L-J parameters for a subset of alkanes: We want to seed inference using a simulation at one parameter point, reweight to parameters with an uncertainty region,
# and build a surrogate model over that

#Define datasets used for comparisons (assume we are comparing to density)
thermoml_keys = ['ethane_key']
ethane_dataset = ThermoMLDataset(keys=thermoml_keys)
# Filter the dataset to include only molar heat capacities measured between 130-260 K
ethane_dataset.filter(ePropName='Liquid Density, g/mL') # filter to retain only this property name
ethane_dataset.filter(VariableType='eTemperature', min=130*unit.kelvin, max=260*kelvin) # retain only measurements with `eTemperature` in specified range
#initialize some sort of MCMC object that uses, for example, Lennard-Jones parameters as inputs.


#some function that calculates a posterior probability based on thermo_ml data and simulated values)

def calc_posterior(ethane_dataset,calculated_properties,uncertainty):
    return log_posterior


#SETUP for MCMC

#Set up a trace/logprob trace
trace=np.zeros((simulation_length,np.size(initial_conditions))
logp_trace=np.zeros(simulation_length)

trace[0]=initial_conditions

#Calculate density data at initial conditions by running simulations at specific temperatures with the L-J parameters set to 
property_calculator_ethane=PropertyCalculationRunner(self,ethane_dataset)
values=property_calculator_ethane.run(parameters=trace[0],CalculationFidelity.Simulation)

#Calculate log prob and set logprob trace:
logp_trace[0]=calc_posterior(ethane_dataset,values.values,values.uncertainty)

#choose a bunch of parameter states to reweight to
reweight_states=[states]
reweighted_values_keep=[]

#Only keep the values that are within the uncertainty from the initial value
for state in reweight_states:
    new_values=property_calculator_ethane.run(parameters=state,CalculationFidelity.Reweighting)
    if abs(values.value-new_values.value) <= values.uncertainty:
        reweighted_values_keep.append(new_values)

#Create surrogate model from reweighted values in trusted region
gp_surrogate_model=GaussianProcessModel(reweighted_values_keep,other_params)

#Now do MCMC, using the surrogate model to calculate posteriors
    
def MCMC_Driver(trace,logptrace,property_calculator_ethane,simulation_length,ethane_dataset,other_parameters):
    for i in range(1,np.length(logptrace-1)):
        
        #propose new parameter values with some sort of proposal
        params=propose_parameters(trace[i-1])
        
        #Calculate values for new parameters based on the surrogate model we create earlier
        params_values=property_calculator_ethane.run(parameters=params,surrogateModel=gp_surrogate_model,CalculationFidelity.surrogateModel)
        
        #Calculate Posterior
        logprob_proposed=calc_posterior(ethane_dataset,params_values.values,params_values.uncertainty)
        
        #Accept or reject new parameter values
        logptrace_new,trace_new=accept_reject(logprob_proposed)
    
        trace[i]=trace_new
        logptrace[i]=logptrace_new
    
    
    

mcmc_bayes_ethane=MCMC_Driver(trace,logptrace,property_calculator_ethane,simulation_length,ethane_dataset,other_parameters)

