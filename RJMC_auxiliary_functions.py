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
from scipy.optimize import minimize,curve_fit
import random as rm
from pymc3.stats import hpd
import matplotlib.patches as mpatches
from datetime import datetime,date
import copy
import math


def computePercentDeviations(compound_2CLJ,temp_values_rhol,temp_values_psat,temp_values_surftens,parameter_values,rhol_data,psat_data,surftens_data,T_c_data,rhol_hat_models,Psat_hat_models,SurfTens_hat_models,T_c_hat_models):
    
    
    rhol_model=rhol_hat_models(compound_2CLJ,temp_values_rhol,*parameter_values)
    psat_model=Psat_hat_models(compound_2CLJ,temp_values_psat,*parameter_values)
    if len(surftens_data) != 0:    
        surftens_model=SurfTens_hat_models(compound_2CLJ,temp_values_surftens,*parameter_values)
        surftens_deviation_vector=((surftens_data-surftens_model)/surftens_data)**2
        surftens_mean_relative_deviation=np.sqrt(np.sum(surftens_deviation_vector)/np.size(surftens_deviation_vector))*100
    else:
        surftens_mean_relative_deviation=0
    T_c_model=T_c_hat_models(compound_2CLJ,*parameter_values)
    
    rhol_deviation_vector=((rhol_data-rhol_model)/rhol_data)**2
    psat_deviation_vector=((psat_data-psat_model)/psat_data)**2
    
    T_c_relative_deviation=(T_c_data-T_c_model)*100/T_c_data
    

    rhol_mean_relative_deviation=np.sqrt(np.sum(rhol_deviation_vector)/np.size(rhol_deviation_vector))*100
    psat_mean_relative_deviation=np.sqrt(np.sum(psat_deviation_vector)/np.size(psat_deviation_vector))*100
    
    
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
        yfile = yaml.load(yfile)#,Loader=yaml.FullLoader)
    ff_params=[]
    params=['eps_lit','sig_lit','Lbond_lit','Q_lit']
    for name in params:
        ff_params.append(yfile["force_field_params"][name])
    
    ff_params_ref=np.transpose(np.asarray(ff_params))
    ff_params_ref[:,1:]=ff_params_ref[:,1:]/10

    Tc_lit = np.loadtxt('TRC_data/'+compound+'/Tc.txt',skiprows=1)
    M_w = np.loadtxt('TRC_data/'+compound+'/Mw.txt',skiprows=1)
    
    df=pd.read_csv('NIST_bondlengths/NIST_bondlengths.txt',delimiter='\t')
    df=df[df.Compound==compound]
    NIST_bondlength=np.asarray(df)
    
    
    data=['rhoL','Pv','SurfTens']
    data_dict={}
    for name in data:
        df=pd.read_csv('TRC_data/'+compound+'/'+name+'.txt',sep='\t')
        df=df.dropna()
        data_dict[name]=df
    return ff_params_ref, Tc_lit, M_w,data_dict, NIST_bondlength[0][1]/10


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

'''
def uncertainty_models(T,T_c,thermo_property):
    Tr=T/T_c
    u=np.zeros(np.size(Tr))
    
    #Linear models for uncertainties in the 2CLJQ correlation we are using, determined from Messerly analysis of figure from Stobener, Stoll, Werth
    
    #Starts at 0.3% for low values and ramps up to 1% for large values    
    if thermo_property == 'rhoL':
        for i in range(np.size(Tr)):
            if Tr[i] < 0.9:
                u[i]=1
            elif 0.9 <= Tr[i] <= 0.95:
                u[i]=1+(3-1)*(Tr[i]-0.9)/(0.95-0.9)
            else:
                u[i]=3
                
    #Starts at 20% for low values and ramps down to 2% for large values
    if thermo_property == 'Pv':
        for i in range(np.size(Tr)):
            if Tr[i] <= 0.55:
                u[i]=20
            elif 0.55 <= Tr[i] <= 0.7:
                u[i]=20+(5-20)*(Tr[i]-0.55)/(0.7-0.55)
            else:
                u[i]=5.0
                
    #Starts at 4% for low values and ramps up to 12% for higher values
    if thermo_property == 'SurfTens':
        for i in range(np.size(Tr)):
            if Tr[i] <= 0.75:
                u[i]=5
            elif 0.75 <= Tr[i] <= 0.95:
                u[i]=10+(20-10)*(Tr[i]-0.75)/(0.95-0.75)
            else:
                u[i]=15
    u/=100
    return u
'''        
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
        

def create_param_triangle_plot_4D(trace,tracename,lit_values,properties,compound,n_iter,file_loc=None):#,sig_prior,eps_prior,L_prior,Q_prior):
    if np.shape(trace) != (0,):
    
        fig,axs=plt.subplots(4,4,figsize=(8,8))
        fig.suptitle('Parameter Marginal Distributions, '+compound+', '+properties+', '+str(n_iter)+' steps',fontsize=20)
        
        axs[0,0].hist(trace[:,1],bins=50,color='m',density=True,label='RJMC Sampling')
        axs[1,1].hist(trace[:,2],bins=50,color='m',density=True)
        axs[2,2].hist(trace[:,3],bins=50,color='m',density=True)
        axs[3,3].hist(trace[:,4],bins=50,color='m',density=True)
        
        '''
        sig_prior=np.multiply(sig_prior,10)
        L_prior=np.multiply(L_prior,10)
        Q_prior=np.multiply(Q_prior,10)
        
        sig_range=np.linspace(0.5*min(trace[:,1]),2*max(trace[:,1]),num=100)
        eps_range=np.linspace(0.5*min(trace[:,2]),2*max(trace[:,2]),num=100)
        L_range=np.linspace(0.5*min(trace[:,3]),2*max(trace[:,3]),num=100)
        
        logitpdf=distributions.logistic.pdf
        '''
        #axs[0,0].plot(sig_range,1000000000*logitpdf(sig_range,*sig_prior))
        #axs[1,1].plot(eps_range,1000000*logitpdf(eps_range,*eps_prior))
        #axs[2,2].plot(L_range,10*logitpdf(L_range,*L_prior))
        
        '''
        axs[0,0].axvline(x=eps_prior[0],color='r',linestyle='--',label='Uniform Prior')
        axs[0,0].axvline(x=eps_prior[1],color='r',linestyle='--')
        axs[1,1].axvline(x=sig_prior[0],color='r',linestyle='--')
        axs[1,1].axvline(x=sig_prior[1],color='r',linestyle='--')
        axs[2,2].axvline(x=L_prior[0],color='r',linestyle='--')
        axs[2,2].axvline(x=L_prior[1],color='r',linestyle='--')
        '''
        #axs[3,3].axvline(x=Q_prior[0],color='r',linestyle='--')
        #axs[3,3].axvline(x=Q_prior[1],color='r',linestyle='--')
        
        

        
        axs[0,1].hist2d(trace[:,2],trace[:,1],bins=100,cmap='cool',label='RJMC Sampling')
        axs[0,2].hist2d(trace[:,3],trace[:,1],bins=100,cmap='cool')
        axs[0,3].hist2d(trace[:,4],trace[:,1],bins=100,cmap='cool')
        axs[1,2].hist2d(trace[:,3],trace[:,2],bins=100,cmap='cool')
        axs[1,3].hist2d(trace[:,4],trace[:,2],bins=100,cmap='cool')
        axs[2,3].hist2d(trace[:,4],trace[:,3],bins=100,cmap='cool')
        
        
        axs[0,1].scatter(lit_values[::4,1],lit_values[::4,0],color='0.25',marker='o',alpha=0.5,facecolors='none',label='Pareto Values')
        axs[0,2].scatter(lit_values[::4,2],lit_values[::4,0],color='0.25',marker='o',alpha=0.5,facecolors='none')
        axs[0,3].scatter(lit_values[::4,3],lit_values[::4,0],color='0.25',marker='o',alpha=0.5,facecolors='none')
        axs[1,2].scatter(lit_values[::4,2],lit_values[::4,1],color='0.25',marker='o',alpha=0.5,facecolors='none')
        axs[1,3].scatter(lit_values[::4,3],lit_values[::4,1],color='0.25',marker='o',alpha=0.5,facecolors='none')
        axs[2,3].scatter(lit_values[::4,3],lit_values[::4,2],color='0.25',marker='o',alpha=0.5,facecolors='none')   
        
       
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
        
    
        axs[0,0].set_ylabel(r'$\epsilon$ (K)',fontsize=14)
        axs[1,1].set_ylabel(r'$\sigma$ ($\AA$)',fontsize=14)
        axs[2,2].set_ylabel(r'L ($\AA$)',fontsize=14)
        axs[3,3].set_ylabel(r'Q (D$\AA$)',fontsize=14)
    
        axs[0,0].set_xlabel(r'$\epsilon$ (K)',fontsize=14) 
        axs[0,1].set_xlabel(r'$\sigma$ ($\AA$)',fontsize=14)
        axs[0,2].set_xlabel(r'L ($\AA$)',fontsize=14)
        axs[0,3].set_xlabel(r'Q (D$\AA$)',fontsize=14)
    
        axs[0,0].xaxis.set_label_position('top')
        axs[0,1].xaxis.set_label_position('top')
        axs[0,2].xaxis.set_label_position('top')
        axs[0,3].xaxis.set_label_position('top')
        
        handles,labels = axs[0,1].get_legend_handles_labels()
        handles0,labels0 = axs[0,0].get_legend_handles_labels()
        #plt.figlegend((label0,label1),('Literature','RJMC Sampling'))
        fig.legend(handles,labels,loc=[0.1,0.4])
        plt.savefig(file_loc+tracename+'.png')
        plt.close()
        #plt.show()
    return
        
        
        
def create_percent_dev_triangle_plot(trace,tracename,lit_values,properties,compound,n_iter,file_loc=None):
    fig,axs=plt.subplots(4,4,figsize=(8,8))
    fig.suptitle('Percent Deviation Marginal Distributions, '+compound+', '+properties+', '+str(n_iter)+' steps')
    axs[0,0].hist(trace[:,0],bins=50,color='m',density=True)
    axs[1,1].hist(trace[:,1],bins=50,color='m',density=True)
    axs[2,2].hist(trace[:,2],bins=50,color='m',density=True)
    axs[3,3].hist(trace[:,3],bins=50,color='m',density=True)
    
    
 
    
    
    axs[0,1].hist2d(trace[:,1],trace[:,0],bins=100,cmap='cool')
    axs[0,2].hist2d(trace[:,2],trace[:,0],bins=100,cmap='cool')
    axs[0,3].hist2d(trace[:,3],trace[:,0],bins=100,cmap='cool')
    axs[1,2].hist2d(trace[:,2],trace[:,1],bins=100,cmap='cool')
    axs[1,3].hist2d(trace[:,3],trace[:,1],bins=100,cmap='cool')
    axs[2,3].hist2d(trace[:,3],trace[:,2],bins=100,cmap='cool')
    
    axs[0,1].scatter(lit_values[::4,1],lit_values[::4,0],color='0.25',marker='o',alpha=0.5,facecolors='none',label='Stobener Pareto Values')
    axs[0,2].scatter(lit_values[::4,2],lit_values[::4,0],color='0.25',marker='o',alpha=0.5,facecolors='none')
    axs[0,3].scatter(lit_values[::4,3],lit_values[::4,0],color='0.25',marker='o',alpha=0.5,facecolors='none')
    axs[1,2].scatter(lit_values[::4,2],lit_values[::4,1],color='0.25',marker='o',alpha=0.5,facecolors='none')
    axs[1,3].scatter(lit_values[::4,3],lit_values[::4,1],color='0.25',marker='o',alpha=0.5,facecolors='none')
    axs[2,3].scatter(lit_values[::4,3],lit_values[::4,2],color='0.25',marker='o',alpha=0.5,facecolors='none')   
    
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
    
    plt.savefig(file_loc+tracename+'.png')
    plt.close()
    #plt.show()


def import_literature_values(criteria,compound):
    df=pd.read_csv('Literature/Pareto_Hasse_'+criteria+'_criteria.txt',delimiter=' ',skiprows=2,usecols=[0,1,2,3,4,5,6,7,8])
    
    df=df[df.Substance==compound]
    df1=df.iloc[:,1:5]
    df2=df.iloc[:,5:9]
    df1=df1[['epsilon','sigma','L','Q']]
    
    return np.asarray(df1),np.asarray(df2)
    #return df1,df2
    
def plot_bar_chart(prob,properties,compound,n_iter,n_models,file_loc=None):
    x=np.arange(n_models)
    prob=prob[-1:]+prob[:-1]
    print(prob)
    prob_copy=copy.deepcopy(prob)
    basis=min(i for i in prob if i > 0)
    #while basis==0:
        #prob_copy=np.delete(prob_copy,np.argmin(prob))
        #if len(prob_copy)==0:
        #    basis=1
        #else:
        #    basis=min(prob_copy)
    value=prob/basis
    if np.size(prob) == 2:
        color=['red','blue']
        label=('AUA,AUA+Q')
    elif np.size(prob) == 3:
        color=['red','blue','orange']
        label=('UA','AUA','AUA+Q')
    plt.bar(x,value,color=['red','blue','orange'])
    plt.xticks(x,('UA','AUA','AUA+Q'),fontsize=14)
    plt.title('Model Bayes Factor, '+compound+', '+properties+', '+str(n_iter)+' steps',fontsize=14)
    plt.ylabel('Bayes Factor',fontsize=14)
    
    plt.savefig(file_loc+'/bar_chart.png')
    plt.close()
    #plt.show()
    return

def recompute_lit_percent_devs(lit_values,computePercentDeviations,temp_values_rhol,temp_values_psat,temp_values_surftens,parameter_values,rhol_data,psat_data,surftens_data,T_c_data,rhol_hat_models,Psat_hat_models,SurfTens_hat_models,T_c_hat_models,compound_2CLJ):
    new_lit_devs=[]
    
    df=pd.DataFrame(lit_values)
    df[4]=1
    cols=[4,0,1,2,3]
    df=df[cols]
    new_lit_values=np.asarray(df)
    new_lit_values[:,2:]/=10
    #print(new_lit_values)
    
    for i in range(np.size(new_lit_values,0)):
        devs=computePercentDeviations(compound_2CLJ,temp_values_rhol,temp_values_psat,temp_values_surftens,new_lit_values[i],rhol_data,psat_data,surftens_data,T_c_data,rhol_hat_models,Psat_hat_models,SurfTens_hat_models,T_c_hat_models)
        new_lit_devs.append(devs)
    return np.asarray(new_lit_devs)
    

def get_metadata(directory,label,compound,properties,sig_prior,eps_prior,L_prior,Q_prior,n_iter,swap_freq,n_points,transition_matrix,prob,attempt_matrix,acceptance_matrix):
    metadata={'compound':compound,'Sigma Prior':sig_prior,'eps_prior':eps_prior,'L_prior':L_prior,'Q_Prior':Q_prior,'MCMC Steps': str(n_iter),'Swap Freq': str(swap_freq),'n_points':str(n_points),'timestamp':str(datetime.today()),
    'Transition Matrix':transition_matrix,'Model Probability':prob,'Attempt Matrix':attempt_matrix,'Acceptance Matrix':acceptance_matrix}
    fname=compound+'_'+properties+'_'+str(n_points)+'_'+str(n_iter)+'_'+str(date.today())+'_'+label
    f=open(directory+'/metadata/'+fname+'.txt',"w")
    f.write( str(metadata) )
    f.close()
    return    



# Create functions that return properties for a given model, eps, sig

def rhol_hat_models(compound_2CLJ,Temp,model,eps,sig,L,Q):
    '''
    L_nm=L/10
    sig_nm=sig/10
    Q_nm=Q/10
    '''
    if model == 0: #Two center AUA LJ
    
        rhol_hat = compound_2CLJ.rhol_hat_2CLJQ(Temp,eps,sig,L,0) 
        
    elif model == 1: #Two center AUA LJ+Q
    
        rhol_hat = compound_2CLJ.rhol_hat_2CLJQ(Temp,eps,sig,L,Q) 
    

    elif model == 2: #2CLJ model
    
        rhol_hat = compound_2CLJ.rhol_hat_2CLJQ(Temp,eps,sig,L,0) 
        
    return rhol_hat #[kg/m3]       
  
def Psat_hat_models(compound_2CLJ,Temp,model,eps,sig,L,Q):
    '''
    L_nm=L/10
    sig_nm=sig/10
    Q_nm=Q/10
    '''
    if model == 0: #Two center AUA LJ
    
        Psat_hat = compound_2CLJ.Psat_hat_2CLJQ(Temp,eps,sig,L,0) 
        
    elif model == 1: #Two center AUA LJ+Q
    
        Psat_hat = compound_2CLJ.Psat_hat_2CLJQ(Temp,eps,sig,L,Q) 
    

    elif model == 2: #2CLJ model
    
        Psat_hat = compound_2CLJ.Psat_hat_2CLJQ(Temp,eps,sig,L,0) 
        
    return Psat_hat #[kPa]       

def SurfTens_hat_models(compound_2CLJ,Temp,model,eps,sig,L,Q):
    '''
    L_nm=L/10
    sig_nm=sig/10
    Q_nm=Q/10
    '''
    if model == 0:
        
        SurfTens_hat=compound_2CLJ.ST_hat_2CLJQ(Temp,eps,sig,L,0)
        
    elif model == 1:
        
        
        
        SurfTens_hat=compound_2CLJ.ST_hat_2CLJQ(Temp,eps,sig,L,Q)
        
    elif model == 2:
        
        #Model 2 is the same as model 0, but the L value will be previously specified (not varying)
        
        SurfTens_hat=compound_2CLJ.ST_hat_2CLJQ(Temp,eps,sig,L,0)
        
    return SurfTens_hat

def T_c_hat_models(compound_2CLJ,model,eps,sig,L,Q):
    '''
    L_nm=L/10
    sig_nm=sig/10
    Q_nm=Q/10
    '''
    if model == 0: 
        
        T_c_hat=compound_2CLJ.T_c_hat_2CLJQ(eps,sig,L,0)
    
    elif model == 1: 
        
        T_c_hat=compound_2CLJ.T_c_hat_2CLJQ(eps,sig,L,Q)
        
    elif model == 2: 
        
        T_c_hat=compound_2CLJ.T_c_hat_2CLJQ(eps,sig,L,0)
        
    return T_c_hat







#parameter_prior_proposals,trace_tuned=mcmc_prior_proposal(n_models,calc_posterior,guess_params,guess_sd)

def calc_posterior_refined(model,eps,sig,L,Q):

    logp = 0
#    print(eps,sig)
    # Using noninformative priors
 
    
    if model == 0:
        Q=0
        logp += dnorm(eps,*parameter_prior_proposals[0,1])
        logp += dnorm(sig,*parameter_prior_proposals[0,2])
    
    
    if model == 1:
        logp += dnorm(eps,*parameter_prior_proposals[1,1])
        logp += dnorm(sig,*parameter_prior_proposals[1,2])
        logp += dnorm(L,*parameter_prior_proposals[1,3])
        logp += dnorm(Q,*parameter_prior_proposals[1,4])
    # OCM: no reason to use anything but uniform priors at this point.  Could probably narrow the prior ranges a little bit to improve acceptance,
    #But Rich is rightly being conservative here especially since evaluations are cheap.
    
#    print(eps,sig)
    #rhol_hat_fake = rhol_hat_models(T_lin,model,eps,sig)
    rhol_hat = rhol_hat_models(T_rhol_data,model,eps,sig,L,Q) #[kg/m3]
    Psat_hat = Psat_hat_models(T_Psat_data,model,eps,sig,L,Q) #[kPa]        
 
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

def fit_exponential(trace,bins=25):
    y,x=np.histogram(trace,bins=bins,density=True)
    x_adjust=[]
    for i in range(len(x)-1):
        x_adjust.append((x[i]+x[i+1])/2)
    def func(x,a,b):
        return a*np.exp(-np.multiply(1/b,x))
    
    popt,pcov= curve_fit(func,x_adjust,y,bounds=(0,[500,400]))
    #plt.plot(x_adjust,func(x_adjust,*popt))
    #plt.plot(x_adjust,y)
    #plt.show()
    return popt

def fit_gamma(trace,bins=25):
    y,x=np.histogram(trace,bins=bins,density=True)
    x_adjust=[]
    for i in range(len(x)-1):
        x_adjust.append((x[i]+x[i+1])/2)
    def func(x,a,b):
        return (1/(sp.special.gamma(a)*(b**a)))*np.power(x,a-1)*np.exp(-x/b)
    
    popt,pcov= curve_fit(func,x_adjust,y,bounds=(0,[500,400]))
    plt.plot(x_adjust,func(x_adjust,*popt))
    plt.plot(x_adjust,y)
    plt.show()
    return popt

def plot_BAR_values(BAR_trace):
    BAR_vector_0_1 = []
    BAR_vector_1_0 = []
    BAR_vector_2_0 = []
    BAR_vector_0_2 = []

    for i in range(len(BAR_trace)):
        if BAR_trace[i, 0] == 0 and BAR_trace[i, 1] == 1:
            if str(BAR_trace[i, 2]) != 'nan':
                BAR_vector_0_1.append(BAR_trace[i, 2])
        elif BAR_trace[i, 0] == 1 and BAR_trace[i,1] == 0:
            if str(BAR_trace[i, 2]) != 'nan':
                BAR_vector_1_0.append(BAR_trace[i, 2])
        elif BAR_trace[i, 0] == 0 and BAR_trace[i,1] == 2:
            if str(BAR_trace[i, 2]) != 'nan':
                BAR_vector_0_2.append(BAR_trace[i, 2])
        elif BAR_trace[i, 0] == 2 and BAR_trace[i,1] == 0:
            if str(BAR_trace[i, 2]) != 'nan':
                BAR_vector_2_0.append(BAR_trace[i, 2])

    print(len(BAR_vector_0_1))
    print(len(BAR_vector_1_0))
    print(len(BAR_vector_0_2))
    print(len(BAR_vector_2_0))
   
    if len(BAR_vector_0_1) != 0 and len(BAR_vector_1_0) != 0:
    
        plt.hist(BAR_vector_0_1,label = '1 --> 0',alpha=0.5,bins=50,density=True)
        plt.xlabel('"Free Energy" Difference')
        plt.ylabel('Frequency')
        plt.title('BAR Histograms, Attempts from 1-->0')
        plt.legend()
        plt.show()
        plt.hist(BAR_vector_1_0,label = '0 --> 1',range=(-10,10000),alpha=0.5,bins=50,density=True)
        plt.xlabel('"Free Energy" Difference')
        plt.ylabel('Frequency')
        plt.title('BAR Histograms,Attempts from 0 --> 1')
        plt.legend()
        plt.show()
            
        
    if len(BAR_vector_0_2) != 0 and len(BAR_vector_2_0) != 0:
        plt.hist(BAR_vector_0_2,label = '2 --> 0', alpha=0.5,bins=50)
        plt.xlabel('"Free Energy" Difference')
        plt.ylabel('Frequency')
        plt.title('BAR Histograms,Attempts from 0 --> 2')
        plt.legend()
        plt.show()
        plt.hist(BAR_vector_2_0,label = '0 --> 2', alpha=0.5,bins=50)
        plt.xlabel('"Free Energy" Difference')
        plt.ylabel('Frequency')
        plt.title('BAR Histograms,Attempts from 2 --> 0')
        plt.legend()
        plt.show()
        
    return BAR_vector_0_1,BAR_vector_1_0



def unbias_simulation(biasing_factor,probabilities):
    unbias_prob = probabilities*np.exp(-biasing_factor)
    unbias_prob_normalized = unbias_prob/sum(unbias_prob)
        
    return unbias_prob_normalized

def undo_bar(BAR_Output):
    for value in BAR_Output:
        if value == 'No Bar Estimate':
            value[0] = 0
            value.append([0,0])
    unnorm_prob = np.asarray([1,1/BAR_Output[0],1/BAR_Output[1]])
    norm_prob = unnorm_prob/sum(unnorm_prob)
    return norm_prob