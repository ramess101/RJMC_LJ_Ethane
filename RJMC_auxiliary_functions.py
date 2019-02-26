#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 11:08:09 2019

@author: owenmadin
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
from pymc3.stats import hpd
import matplotlib.patches as mpatches
from datetime import datetime,date
'''
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

T_min = 167.9
T_max = 290.1

rhol_data = rhol_data[T_rhol_data>T_min]
T_rhol_data = T_rhol_data[T_rhol_data>T_min]
rhol_data = rhol_data[T_rhol_data<T_max]
T_rhol_data = T_rhol_data[T_rhol_data<T_max]

Psat_data = Psat_data[T_Psat_data>T_min]
T_Psat_data = T_Psat_data[T_Psat_data>T_min]
Psat_data = Psat_data[T_Psat_data<T_max]
T_Psat_data = T_Psat_data[T_Psat_data<T_max]
'''
def computePercentDeviations(temp_values_rhol,temp_values_psat,temp_values_surftens,parameter_values,rhol_data,psat_data,surftens_data,T_c_data,rhol_hat_models,Psat_hat_models,SurfTens_hat_models,T_c_hat_models):
    
    
    rhol_model=rhol_hat_models(temp_values_rhol,*parameter_values)
    psat_model=Psat_hat_models(temp_values_psat,*parameter_values)
    surftens_model=SurfTens_hat_models(temp_values_surftens,*parameter_values)
    T_c_model=T_c_hat_models(*parameter_values)
    
    rhol_deviation_vector=((rhol_data-rhol_model)/rhol_data)**2
    psat_deviation_vector=((psat_data-psat_model)/psat_data)**2
    surftens_deviation_vector=((surftens_data-surftens_model)/surftens_data)**2
    T_c_relative_deviation=(T_c_data-T_c_model)*100/T_c_data
    

    rhol_mean_relative_deviation=np.sqrt(np.sum(rhol_deviation_vector)/np.size(rhol_deviation_vector))*100
    psat_mean_relative_deviation=np.sqrt(np.sum(psat_deviation_vector)/np.size(psat_deviation_vector))*100
    surftens_mean_relative_deviation=np.sqrt(np.sum(surftens_deviation_vector)/np.size(surftens_deviation_vector))*100
    
    return rhol_mean_relative_deviation, psat_mean_relative_deviation, surftens_mean_relative_deviation, T_c_relative_deviation
    
        
def plotPercentDeviations(percent_deviation_trace,max_apd,label1,label2):
    
    plt.scatter(percent_deviation_trace[:,0],percent_deviation_trace[:,1],alpha=0.5,marker='x',label=label1)
    plt.scatter(max_apd[:,0],max_apd[:,1],alpha=1,marker='x',color='r',label=label2)
    plt.scatter(percent_deviation_trace[0,0],percent_deviation_trace[0,1],alpha=1,marker='x',color='orange',label='Literature')
    plt.xlabel(r'% Deviation, $\rho_l$')
    plt.ylabel(r'% Deviation, $P_{sat}$')
    plt.gca().set_xlim(left=0)
    plt.gca().set_ylim(bottom=0)
    plt.legend()
    plt.show()
    
    plt.scatter(percent_deviation_trace[:,0],percent_deviation_trace[:,2],alpha=0.5,marker='x',label=label1)
    plt.scatter(max_apd[:,0],max_apd[:,2],alpha=1,marker='x',color='r',label=label2)
    plt.scatter(percent_deviation_trace[0,0],percent_deviation_trace[0,2],alpha=1,marker='x',color='orange',label='Literature')
    plt.xlabel(r'% Deviation, $\rho_l$')
    plt.ylabel(r'% Deviation, $P_{sat}$')
    plt.gca().set_xlim(left=0)
    plt.gca().set_ylim(bottom=0)
    plt.legend()
    plt.show()
    
    plt.scatter(percent_deviation_trace[:,1],percent_deviation_trace[:,2],alpha=0.5,marker='x',label=label1)
    plt.scatter(max_apd[:,1],max_apd[:,2],alpha=1,marker='x',color='r',label=label2)
    plt.scatter(percent_deviation_trace[0,1],percent_deviation_trace[0,2],alpha=1,marker='x',color='orange',label='Literature')
    plt.xlabel(r'% Deviation, $\rho_l$')
    plt.ylabel(r'% Deviation, $P_{sat}$')
    plt.gca().set_xlim(left=0)
    plt.gca().set_ylim(bottom=0)
    plt.legend()
    plt.show()
    return

def plotDeviationHistogram(percent_deviation_trace,pareto_point):
    plt.gca().set_xlim(left=0)
    plt.gca().set_ylim(bottom=0)
    plt.hist2d(percent_deviation_trace[:,0],percent_deviation_trace[:,1],bins=100,range=[[0,np.max(percent_deviation_trace[:,0])],[0,np.max(percent_deviation_trace[:,1])]])
    plt.scatter(pareto_point[:,0],pareto_point[:,1],color='r',marker='.',alpha=0.5)
    plt.scatter(percent_deviation_trace[0,0],percent_deviation_trace[0,1],alpha=1,marker='x',color='orange',label='Literature')
    plt.xlabel(r'% Deviation, $\rho_l$')
    plt.ylabel(r'% Deviation, $P_{sat}$')

    plt.show()
    
    plt.gca().set_xlim(left=0)
    plt.gca().set_ylim(bottom=0)
    plt.hist2d(percent_deviation_trace[:,0],percent_deviation_trace[:,2],bins=100,range=[[0,np.max(percent_deviation_trace[:,0])],[0,np.max(percent_deviation_trace[:,2])]])
    plt.scatter(pareto_point[:,0],pareto_point[:,2],color='r',marker='.',alpha=0.5)
    plt.scatter(percent_deviation_trace[0,0],percent_deviation_trace[0,2],alpha=1,marker='x',color='orange',label='Literature')
    plt.xlabel(r'% Deviation, $\rho_l$')
    plt.ylabel(r'% Deviation, $\gamma$')

    plt.show()
    
    plt.gca().set_xlim(left=0)
    plt.gca().set_ylim(bottom=0)
    plt.hist2d(percent_deviation_trace[:,1],percent_deviation_trace[:,2],bins=100,range=[[0,np.max(percent_deviation_trace[:,1])],[0,np.max(percent_deviation_trace[:,2])]])
    plt.scatter(pareto_point[:,1],pareto_point[:,2],color='r',marker='.',alpha=0.5)
    plt.scatter(percent_deviation_trace[0,1],percent_deviation_trace[0,2],alpha=1,marker='x',color='orange',label='Literature')
    plt.xlabel(r'% Deviation, $P_{sat}$')
    plt.ylabel(r'% Deviation, $\gamma$')

    plt.show()
    
    plt.hist(percent_deviation_trace[:,0],bins=100,density=True)
    plt.xlabel(r'% Deviation, $\rho_l$')
    plt.ylabel('Probability Density')
    plt.show()

    plt.hist(percent_deviation_trace[:,1],bins=100,density=True)
    plt.xlabel(r'% Deviation, $P_{sat}$')
    plt.ylabel('Probability Density')
    plt.show()
    
    return

def findParetoPoints(percent_deviation_trace,trace,tolerance):
    total_percent_dev=np.sum(abs(percent_deviation_trace[:,:2]),1)
    pareto_point=percent_deviation_trace[np.argmin(total_percent_dev)]
    pareto_points=[]
    pareto_points.append(pareto_point)
    pareto_point_values=[]
    pareto_point_values.append(trace[np.argmin(total_percent_dev)])
    '''
    for i in range(np.size(total_percent_dev)):
        if total_percent_dev[i] <= sum(pareto_point)+tolerance:
            pareto_points.append(percent_deviation_trace[i])
            pareto_point_values.append(trace[i])
    '''
    pareto_points=np.asarray(pareto_points)
    pareto_point_values=np.asarray(pareto_point_values)
    return pareto_points,pareto_point_values

def findSingleMinPoints(percent_deviation_trace,trace):
    min_points=[]
    min_points_values=[]
    for i in range(np.size(percent_deviation_trace,1)):
        min_points_values.append(trace[np.argmin(abs(percent_deviation_trace[:,i]))])
        min_points.append(percent_deviation_trace[np.argmin(abs(percent_deviation_trace[:,i]))])
        
    return min_points,min_points_values



def parse_data_ffs(compound):
    fname = "lit_forcefields/"+compound+".yaml"
    with open(fname) as yfile:
        yfile = yaml.load(yfile)
    ff_params=[]
    params=['eps_lit','sig_lit','Lbond_lit','Q_lit']
    for name in params:
        ff_params.append(yfile["force_field_params"][name])
    
    ff_params_ref=np.transpose(np.asarray(ff_params))
    ff_params_ref[:,1:]=ff_params_ref[:,1:]/10

    Tc_lit = np.loadtxt('TRC_data/'+compound+'/Tc.txt',skiprows=1)
    M_w = np.loadtxt('TRC_data/'+compound+'/Mw.txt',skiprows=1)
    
    data=['rhoL','Pv','SurfTens']
    data_dict={}
    for name in data:
        df=pd.read_table('TRC_data/'+compound+'/'+name+'.txt')
        df=df.dropna()
        data_dict[name]=df
    return ff_params_ref, Tc_lit, M_w,data_dict


def filter_thermo_data(thermo_data,T_min,T_max,n_points):
    for name in thermo_data:
        df=thermo_data[name]
        

        df=df[df.values[:,0]>T_min]
        df=df[df.values[:,0]<T_max]
        if int(np.floor(df.shape[0]/(n_points-1))) == 0:
            slicer=1
        else:
            slicer=int(np.floor(df.shape[0]/(n_points-1)))
        #print(slicer)
        df=df[::slicer]
        thermo_data[name]=df
        
    return thermo_data

def uncertainty_models(T,T_c,thermo_property):
    Tr=T/T_c
    u=np.zeros(np.size(Tr))
    
    #Linear models for uncertainties in the 2CLJQ correlation we are using, determined from Messerly analysis of figure from Stobener, Stoll, Werth
    
    #Starts at 0.3% for low values and ramps up to 1% for large values    
    if thermo_property == 'rhoL':
        for i in range(np.size(Tr)):
            if Tr[i] < 0.9:
                u[i]=0.3
            elif 0.9 <= Tr[i] <= 0.95:
                u[i]=0.3+(1-0.3)*(Tr[i]-0.9)/(0.95-0.9)
            else:
                u[i]=1.0
                
    #Starts at 20% for low values and ramps down to 2% for large values
    if thermo_property == 'Pv':
        for i in range(np.size(Tr)):
            if Tr[i] <= 0.55:
                u[i]=20
            elif 0.55 <= Tr[i] <= 0.7:
                u[i]=20+(2-20)*(Tr[i]-0.55)/(0.7-0.55)
            else:
                u[i]=2.0
                
    #Starts at 4% for low values and ramps up to 12% for higher values
    if thermo_property == 'SurfTens':
        for i in range(np.size(Tr)):
            if Tr[i] <= 0.75:
                u[i]=4
            elif 0.75 <= Tr[i] <= 0.95:
                u[i]=4+(12-4)*(Tr[i]-0.75)/(0.95-0.75)
            else:
                u[i]=12.0
    u/=100
    return u
            
def calculate_uncertainties(thermo_data,T_c):
    u_dict={}
    for name in thermo_data:
        
        #Extract data from our data arrays
        data=np.asarray(thermo_data[name])
        T=data[:,0]
        values=data[:,1]
        u_exp=data[:,2]
        
        pu_corr=uncertainty_models(T,T_c,name)
        u_corr=pu_corr*values
        
        u_tot=np.sqrt(u_corr**2+u_exp**2)
        u_dict[name]=u_tot
    return u_dict
        

def create_param_triangle_plot_4D(trace,fname,tracename,lit_values,properties,compound,n_iter):
    if np.shape(trace) != (0,):
    
        fig,axs=plt.subplots(4,4,figsize=(8,8))
        fig.suptitle('Parameter Marginal Distributions, '+compound+', '+properties+', '+str(n_iter)+' steps')
        axs[0,0].hist(trace[:,1],bins=50,color='m',density=True,label='RJMC Sampling')
        axs[1,1].hist(trace[:,2],bins=50,color='m',density=True)
        axs[2,2].hist(trace[:,3],bins=50,color='m',density=True)
        axs[3,3].hist(trace[:,4],bins=50,color='m',density=True)
        
        
        axs[0,1].scatter(lit_values[::4,1],lit_values[::4,0],color='g',marker='+',label='Literature Values')
        axs[0,2].scatter(lit_values[::4,2],lit_values[::4,0],color='g',marker='+')
        axs[0,3].scatter(lit_values[::4,3],lit_values[::4,0],color='g',marker='+')
        axs[1,2].scatter(lit_values[::4,2],lit_values[::4,1],color='g',marker='+')
        axs[1,3].scatter(lit_values[::4,3],lit_values[::4,1],color='g',marker='+')
        axs[2,3].scatter(lit_values[::4,3],lit_values[::4,2],color='g',marker='+')    
        
        axs[0,1].hist2d(trace[:,2],trace[:,1],bins=100,cmap='plasma',label='RJMC Sampling')
        axs[0,2].hist2d(trace[:,3],trace[:,1],bins=100,cmap='plasma')
        axs[0,3].hist2d(trace[:,4],trace[:,1],bins=100,cmap='plasma')
        axs[1,2].hist2d(trace[:,3],trace[:,2],bins=100,cmap='plasma')
        axs[1,3].hist2d(trace[:,4],trace[:,2],bins=100,cmap='plasma')
        axs[2,3].hist2d(trace[:,4],trace[:,3],bins=100,cmap='plasma')
        


       
        #axs[0,1].set_ylim([min(lit_values[:,0]),max(lit_values[:,0])])
        
        
        
        fig.delaxes(axs[1,0])
        fig.delaxes(axs[2,0])
        fig.delaxes(axs[3,0])
        fig.delaxes(axs[2,1])
        fig.delaxes(axs[3,1])
        fig.delaxes(axs[3,2])
        '''
        axs[0,0].axes.get_yaxis().set_visible(False)
        axs[1,1].axes.get_yaxis().set_visible(False)
        axs[2,2].axes.get_yaxis().set_visible(False)
        axs[3,3].axes.get_yaxis().set_visible(False)
        '''
        axs[0,1].axes.get_yaxis().set_visible(False)
        axs[0,2].axes.get_yaxis().set_visible(False)
        axs[1,2].axes.get_yaxis().set_visible(False)
        axs[1,3].axes.get_xaxis().set_visible(False)
        axs[2,3].axes.get_xaxis().set_visible(False)
    
        
        axs[0,0].xaxis.tick_top()
        axs[0,1].xaxis.tick_top()
        axs[0,2].xaxis.tick_top()
        axs[0,3].xaxis.tick_top()
        axs[0,3].yaxis.tick_right()
        axs[1,3].yaxis.tick_right()
        axs[2,3].yaxis.tick_right()
        
        axs[0,0].set_yticklabels([])
        axs[1,1].set_yticklabels([]) 
        axs[2,2].set_yticklabels([]) 
        axs[3,3].set_yticklabels([]) 
        
    
        axs[0,0].set(ylabel=r'$\epsilon$ (K)')
        axs[1,1].set(ylabel=r'$\sigma (\dot{A}$)')
        axs[2,2].set(ylabel=r'L ($\dot{A}$)')
        axs[3,3].set(ylabel=r'Q ($D\dot{A}$)')
    
        axs[0,0].set(xlabel=r'$\epsilon$ (K)') 
        axs[0,1].set(xlabel=r'$\sigma (\dot{A}$)')
        axs[0,2].set(xlabel=r'L ($\dot{A}$)')
        axs[0,3].set(xlabel=r'Q ($D\dot{A}$)')
    
        axs[0,0].xaxis.set_label_position('top')
        axs[0,1].xaxis.set_label_position('top')
        axs[0,2].xaxis.set_label_position('top')
        axs[0,3].xaxis.set_label_position('top')
        
        handles,labels = axs[0,1].get_legend_handles_labels()
        handles0,labels0 = axs[0,0].get_legend_handles_labels()
        #plt.figlegend((label0,label1),('Literature','RJMC Sampling'))
        fig.legend(handles,labels,loc=[0.05,0.3])
        plt.savefig('triangle_plots/'+fname+tracename+'.png')
        plt.show()
    return
        
        
        
def create_percent_dev_triangle_plot(trace,fname,tracename,lit_values,prob,properties,compound,n_iter):
    fig,axs=plt.subplots(4,4,figsize=(8,8))
    fig.suptitle('Percent Deviation Marginal Distributions, '+compound+', '+properties+', '+str(n_iter)+' steps')
    axs[0,0].hist(trace[:,0],bins=50,color='m',density=True)
    axs[1,1].hist(trace[:,1],bins=50,color='m',density=True)
    axs[2,2].hist(trace[:,2],bins=50,color='m',density=True)
    axs[3,3].hist(trace[:,3],bins=50,color='m',density=True)
    
    
    axs[0,1].scatter(lit_values[::4,1],lit_values[::4,0],color='g',marker='+',label='Literature Values')
    axs[0,2].scatter(lit_values[::4,2],lit_values[::4,0],color='g',marker='+')
    axs[0,3].scatter(lit_values[::4,3],lit_values[::4,0],color='g',marker='+')
    axs[1,2].scatter(lit_values[::4,2],lit_values[::4,1],color='g',marker='+')
    axs[1,3].scatter(lit_values[::4,3],lit_values[::4,1],color='g',marker='+')
    axs[2,3].scatter(lit_values[::4,3],lit_values[::4,2],color='g',marker='+')    
    
    
    axs[0,1].hist2d(trace[:,1],trace[:,0],bins=100,cmap='plasma')
    axs[0,2].hist2d(trace[:,2],trace[:,0],bins=100,cmap='plasma')
    axs[0,3].hist2d(trace[:,3],trace[:,0],bins=100,cmap='plasma')
    axs[1,2].hist2d(trace[:,2],trace[:,1],bins=100,cmap='plasma')
    axs[1,3].hist2d(trace[:,3],trace[:,1],bins=100,cmap='plasma')
    axs[2,3].hist2d(trace[:,3],trace[:,2],bins=100,cmap='plasma')
    

    
    #axs[0,1].set_xlim([min(lit_values[::4,1]),max(lit_values[::4,1])])
    #axs[0,1].set_ylim([min(lit_values[::4,0]),max(lit_values[::4,0])])
    

    fig.delaxes(axs[1,0])
    fig.delaxes(axs[2,0])
    fig.delaxes(axs[3,0])
    fig.delaxes(axs[2,1])
    fig.delaxes(axs[3,1])
    fig.delaxes(axs[3,2])
    
    axs[0,1].axes.get_yaxis().set_visible(False)
    axs[0,2].axes.get_yaxis().set_visible(False)
    axs[1,2].axes.get_yaxis().set_visible(False)
    axs[1,3].axes.get_xaxis().set_visible(False)
    axs[2,3].axes.get_xaxis().set_visible(False)

    
    axs[0,0].xaxis.tick_top()
    axs[0,1].xaxis.tick_top()
    axs[0,2].xaxis.tick_top()
    axs[0,3].xaxis.tick_top()
    axs[0,3].yaxis.tick_right()
    axs[1,3].yaxis.tick_right()
    axs[2,3].yaxis.tick_right()
    
    axs[0,0].set_yticklabels([])
    axs[1,1].set_yticklabels([]) 
    axs[2,2].set_yticklabels([]) 
    axs[3,3].set_yticklabels([]) 
    

    axs[0,0].set(ylabel=r'% Deviation, $\rho_l$')
    axs[1,1].set(ylabel=r'% Deviation, $P_{sat}$')
    axs[2,2].set(ylabel=r'% Deviation, $\gamma$')
    axs[3,3].set(ylabel=r'% Deviation, $T_c$')

    axs[0,0].set(xlabel=r'% Deviation, $\rho_l$') 
    axs[0,1].set(xlabel=r'% Deviation, $P_{sat}$')
    axs[0,2].set(xlabel=r'% Deviation, $\gamma$')
    axs[0,3].set(xlabel=r'% Deviation, $T_c$')

    axs[0,0].xaxis.set_label_position('top')
    axs[0,1].xaxis.set_label_position('top')
    axs[0,2].xaxis.set_label_position('top')
    axs[0,3].xaxis.set_label_position('top')
    

    
    
    handles,labels = axs[0,1].get_legend_handles_labels()
    fig.legend(handles,labels,loc=[0.05,0.3])
    
    plt.savefig('triangle_plots/'+fname+tracename+'.png')
    plt.show()


def import_literature_values(criteria,compound):
    df=pd.read_csv('Literature/Pareto_Hasse_'+criteria+'_criteria.txt',delimiter=' ',skiprows=2,usecols=[0,1,2,3,4,5,6,7,8])
    
    df=df[df.Substance==compound]
    df1=df.iloc[:,1:5]
    df2=df.iloc[:,5:9]
    df1=df1[['epsilon','sigma','L','Q']]
    
    return np.asarray(df1),np.asarray(df2)
    #return df1,df2
    
def plot_bar_chart(prob,filename,properties,compound,n_iter):
    x=np.arange(2)
    basis=min(prob)
    value=prob/basis
    plt.bar(x,value,color=['red','blue'])
    plt.xticks(x,('AUA','AUA+Q'))
    plt.title('Model Bayes Factor, '+compound+', '+properties+', '+str(n_iter)+' steps')
    plt.ylabel('Bayes Factor')
    
    plt.savefig('bar_charts/bayes_factor'+filename+'.png')
    plt.show()
    return

def recompute_lit_percent_devs(lit_values,computePercentDeviations,temp_values_rhol,temp_values_psat,temp_values_surftens,parameter_values,rhol_data,psat_data,surftens_data,T_c_data,rhol_hat_models,Psat_hat_models,SurfTens_hat_models,T_c_hat_models):
    new_lit_devs=[]
    
    df=pd.DataFrame(lit_values)
    df[4]=1
    cols=[4,0,1,2,3]
    df=df[cols]
    new_lit_values=np.asarray(df)
    new_lit_values[:,2:]/=10
    #print(new_lit_values)
    
    for i in range(np.size(new_lit_values,0)):
        devs=computePercentDeviations(temp_values_rhol,temp_values_psat,temp_values_surftens,new_lit_values[i],rhol_data,psat_data,surftens_data,T_c_data,rhol_hat_models,Psat_hat_models,SurfTens_hat_models,T_c_hat_models)
        new_lit_devs.append(devs)
    return np.asarray(new_lit_devs)
    

def get_metadata(compound,properties,sig_prior,eps_prior,L_prior,Q_prior,n_iter,swap_freq,n_points,transition_matrix,prob,attempt_matrix,acceptance_matrix):
    metadata={'compound':compound,'Sigma Prior':sig_prior,'eps_prior':eps_prior,'L_prior':L_prior,'Q_Prior':Q_prior,'MCMC Steps': str(n_iter),'Swap Freq': str(swap_freq),'n_points':str(n_points),'timestamp':str(datetime.today()),
    'Transition Matrix':transition_matrix,'Model Probability':prob,'Attempt Matrix':attempt_matrix,'Acceptance Matrix':acceptance_matrix}
    fname=compound+'_'+properties+'_'+str(n_points)+'_'+str(n_iter)+'_'+str(date.today())
    f=open('metadata/'+fname+'.txt',"w")
    f.write( str(metadata) )
    f.close()
    return    