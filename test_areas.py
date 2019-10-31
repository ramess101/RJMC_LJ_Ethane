#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 11:20:11 2019

@author: owenmadin
"""

import numpy as np
import matplotlib.pyplot as plt
import copy
'''
aua_trace = np.load('output/Br2/rhol+Psat/Br2_rhol+Psat_2000000_aua_only_2019-10-29/trace/trace.npy')

auaq_trace = np.load('output/Br2/rhol+Psat/Br2_rhol+Psat_2000000_auaq_only_2019-10-29/trace/trace.npy')

'''
trace = np.load('output/C2H6/rhol+Psat/C2H6_rhol+Psat_1000000_BAR_testing_2019-09-27/trace/trace.npy')

trace_model_0=[]
trace_model_1=[]
trace_model_2=[]
for i in range(len(trace)):
    if trace[i,0] == 0:
        trace_model_0.append(trace[i])
        #log_trace_0.append(logp_trace[i])
    elif trace[i,0] == 1:
        trace_model_1.append(trace[i])
        #log_trace_1.append(logp_trace[i])
    elif trace[i,0] == 2:
        trace_model_2.append(trace[i])
        #log_trace_2.append(logp_trace[i])                
        
aua_trace=np.asarray(trace_model_0)
auaq_trace=np.asarray(trace_model_1)

trace_model_2=np.asarray(trace_model_2)

aua = np.asarray(copy.deepcopy(AUA_params))
auaq = np.asarray(copy.deepcopy(AUAQ_params))

ratio = auaq/aua

key0,values0 = find_maxima(aua_trace)

aua_max_likelihood = values0

key1,values1 = find_maxima(auaq_trace)

auaq_max_likelihood = values1


plot_max_like = True
fig,axs = plt.subplots(4,4,figsize=(15,15))
label_offaxis = False
label_axis = False
for i in range(1,len(aua_trace[0])):
    if label_axis == False:
        axs[i-1,i-1].hist(aua_trace[:,i],density=True,alpha=0.7,bins=50,color='b',label='AUA histogram')
        axs[i-1,i-1].hist(auaq_trace[:,i],density=True,alpha=0.7,bins=50,color = 'orange',label='AUA+Q histogram')
        axs[i-1,i-1].scatter(aua[i-1],0,color = 'b',marker='x',label='AUA "Optimum"')
        axs[i-1,i-1].scatter(auaq[i-1],0,color = 'orange',marker='x',label='AUA+Q "Optimum"')
        if plot_max_like == True:
            axs[i-1,i-1].scatter(aua_max_likelihood[i-1],0,color = 'b',marker='*',label='AUA max likelihood')
            axs[i-1,i-1].scatter(auaq_max_likelihood[i-1],0,color = 'orange',marker='*',label='AUA+Q max likelihood')
        if i < 4:
            axs[i-1,i-1].hist(ratio[i-1]*aua_trace[:,i],density=True,alpha=0.5,bins=50,color='red',label='Transformed AUA histogram')
            axs[i-1,i-1].hist((1/ratio[i-1])*auaq_trace[:,i],density=True,alpha=0.5,bins=50,color='skyblue',label='Transformed AUA+Q histogram')
        label_axis = True
    else:
        axs[i-1,i-1].hist(aua_trace[:,i],density=True,alpha=0.5,bins=50,color='b')
        axs[i-1,i-1].hist(auaq_trace[:,i],density=True,alpha=0.5,bins=50,color = 'orange')
        axs[i-1,i-1].scatter(aua[i-1],0,color = 'b',marker='x')
        axs[i-1,i-1].scatter(auaq[i-1],0,color = 'orange',marker='x')
        if plot_max_like == True:
            axs[i-1,i-1].scatter(aua_max_likelihood[i-1],0,color = 'b',marker='*')
            axs[i-1,i-1].scatter(auaq_max_likelihood[i-1],0,color = 'orange',marker='*')
        if i < 4:
            axs[i-1,i-1].hist(ratio[i-1]*aua_trace[:,i],density=True,alpha=0.5,bins=50,color='red')
            axs[i-1,i-1].hist((1/ratio[i-1])*auaq_trace[:,i],density=True,alpha=0.5,bins=50,color='skyblue')
        
    for j in range(1,len(aua_trace[0])):

        if i==j:
            continue
        elif i>j:
            fig.delaxes(axs[i-1,j-1])
        else:
            if label_offaxis == False: 
                axs[i-1,j-1].scatter(aua_trace[::10,i],aua_trace[::10,j],marker='.',alpha=0.5,label='AUA scatter')
                axs[i-1,j-1].scatter(auaq_trace[::10,i],auaq_trace[::10,j],marker='.',alpha=0.5,label='AUA+Q scatter')
                axs[i-1,j-1].scatter(aua[i-1],aua[j-1],facecolor='b',marker='o',edgecolors='black',label='AUA "Optimum"')
                axs[i-1,j-1].scatter(auaq[i-1],auaq[j-1],color='orange',marker='o',edgecolors='k',label='AUA+Q "Optimum"')
                if plot_max_like == True:
                    axs[i-1,j-1].scatter(aua_max_likelihood[i-1],aua_max_likelihood[j-1],facecolor='b',marker='s',edgecolors='black',label='AUA max likelihood')
                    axs[i-1,j-1].scatter(auaq_max_likelihood[i-1],auaq_max_likelihood[j-1],color='orange',marker='s',edgecolors='k',label='AUA+Q max likelihood')                
                label_offaxis = True
            else:
                axs[i-1,j-1].scatter(aua_trace[::10,i],aua_trace[::10,j],marker='.',alpha=0.5)
                axs[i-1,j-1].scatter(auaq_trace[::10,i],auaq_trace[::10,j],marker='.',alpha=0.5)
                axs[i-1,j-1].scatter(aua[i-1],aua[j-1],color='b',marker='o',edgecolors='k')
                axs[i-1,j-1].scatter(auaq[i-1],auaq[j-1],color='orange',marker='o',edgecolors='k')
                if plot_max_like == True:
                    axs[i-1,j-1].scatter(aua_max_likelihood[i-1],aua_max_likelihood[j-1],facecolor='b',marker='s',edgecolors='black')
                    axs[i-1,j-1].scatter(auaq_max_likelihood[i-1],auaq_max_likelihood[j-1],color='orange',marker='s',edgecolors='k')                
                

axs[0,3].set_ylabel(r'$\epsilon$ (K)',fontsize=14)
axs[1,3].set_ylabel(r'$\sigma$ ($\AA$)',fontsize=14)
axs[2,3].set_ylabel(r'L ($\AA$)',fontsize=14)
axs[3,3].set_ylabel(r'Q (D$\AA$)',fontsize=14)

axs[0,0].set_xlabel(r'$\epsilon$ (K)',fontsize=14) 
axs[0,1].set_xlabel(r'$\sigma$ ($\AA$)',fontsize=14)
axs[0,2].set_xlabel(r'L ($\AA$)',fontsize=14)
axs[0,3].set_xlabel(r'Q (D$\AA$)',fontsize=14)


axs[0,3].yaxis.set_label_position('right')
axs[1,3].yaxis.set_label_position('right')
axs[2,3].yaxis.set_label_position('right')
axs[3,3].yaxis.set_label_position('right')



axs[0,0].xaxis.set_label_position('top')
axs[0,1].xaxis.set_label_position('top')
axs[0,2].xaxis.set_label_position('top')
axs[0,3].xaxis.set_label_position('top')
fig.legend(loc=[0.1,0.2])
fig.suptitle('Br2, rhol+Psat, AUA/AUA+Q Overlap',fontsize=24)
fig.show()

