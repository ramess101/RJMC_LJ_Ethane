from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import scipy as sp

# Conversion constants

k_B = 1.38065e-23 #[J/K]
N_A = 6.02214e23 #[1/mol]
m3_to_nm3 = 1e27
gm_to_kg = 1./1000
J_to_kJ = 1./1000
J_per_m3_to_kPA = 1./1000

class LennardJones_2C():
    def __init__(self,M_w):
        
        self.M_w = M_w
        
        with open("DCLJQ_fluid.yaml") as yfile:
            yfile = yaml.load(yfile)
    
        self.T_c_star_params = np.array(yfile["correlation_parameters"]["Stoll"]["T_c_star_params"])
        self.rho_c_star_params = np.array(yfile["correlation_parameters"]["Stoll"]["rho_c_star_params"])
        self.b_C1 = np.array(yfile["correlation_parameters"]["Stoll"]["rho_L_star_params"]["C1_params"])
        self.b_C2_L = np.array(yfile["correlation_parameters"]["Stoll"]["rho_L_star_params"]["C2_params"])
        self.b_C3_L = np.array(yfile["correlation_parameters"]["Stoll"]["rho_L_star_params"]["C3_params"])
        self.b_C2_v = np.array(yfile["correlation_parameters"]["Stoll"]["rho_v_star_params"]["C2_params"])
        self.b_C3_v = np.array(yfile["correlation_parameters"]["Stoll"]["rho_v_star_params"]["C3_params"])
        self.b_c1 = np.array(yfile["correlation_parameters"]["Stoll"]["P_v_star_params"]["c1_params"])
        self.b_c2 = np.array(yfile["correlation_parameters"]["Stoll"]["P_v_star_params"]["c2_params"])
        self.b_c3 = np.array(yfile["correlation_parameters"]["Stoll"]["P_v_star_params"]["c3_params"])
    
    def T_c_star_hat(self,q,l):
        b=self.T_c_star_params
        x = np.array([1, q**2, q**3, 1./(0.1+l**2), 1./(0.1+l**5), q**2/(0.1+l**2), q**2/(0.1+l**5), q**3/(0.1+l**2), q**3/(0.1+l**5)])
        T_c_star = x*b
        T_c_star = T_c_star.sum()
        return T_c_star
    
    def rho_c_star_hat(self,q,l):
        b=self.rho_c_star_params
        x = np.array([1, q**2, q**3, l**2/(0.11+l**2), l**5/(0.11+l**5), l**2*q**2/(0.11+l**2), l**5*q**2/(0.11+l**5), l**2*q**3/(0.11+l**2), l**5*q**3/(0.11+l**5)])
        rho_c_star = x*b
        rho_c_star = rho_c_star.sum()
        return rho_c_star
    
    def C1_hat(self,q,l,b):
        x_C1 = np.array([1, q**2, q**3, l**3/(l+0.4)**3, l**4/(l+0.4)**5, q**2*l**2/(l+0.4), q**2*l**3/(l+0.4)**7, q**3*l**2/(l+0.4), q**3*l**3/(l+0.4)**7])
        C1 = x_C1*b
        C1 = C1.sum()
        return C1
    
    def C2_hat(self,q,l,b):
        x_C2 = np.array([1, q**2, q**3, l**2, l**3, q**2*l**2, q**2*l**3, q**3*l**2])
        C2 = x_C2*b
        C2 = C2.sum()
        return C2
    
    def C3_hat(self,q,l,b):
        x_C3 = np.array([1, q**2, q**3, l, l**4, q**2*l, q**2*l**4, q**3*l**4])
        C3 = x_C3*b
        C3 = C3.sum()
        return C3
    
    def rho_star_hat_2CLJQ(self,T_star,q,l,phase):   
        b_C1, b_C2_L, b_C3_L, b_C2_v, b_C3_v = self.b_C1, self.b_C2_L, self.b_C3_L, self.b_C2_v, self.b_C3_v
        T_c_star = self.T_c_star_hat(q,l)
        rho_c_star = self.rho_c_star_hat(q,l)
        tau = T_c_star - T_star # T_c_star - T_star  
        if all(tau>0):
            x = np.ones([len(tau),4]) # First column is supposed to be all ones
            x[:,1] = tau**(1./3)
            x[:,2] = tau
            x[:,3] = tau**(3./2)
            C1 = self.C1_hat(q,l,b_C1)
            if phase == 'liquid':
                C2 = self.C2_hat(q,l,b_C2_L)
                C3 = self.C3_hat(q,l,b_C3_L)
                b = np.array([rho_c_star, C1, C2, C3])
            elif phase == 'vapor':
                C2 = self.C2_hat(q,l,b_C2_v)
                C3 = self.C3_hat(q,l,b_C3_v)
                b = np.array([rho_c_star, -C1, C2, C3])
            else:
                return 0
            #rho_star = b[0]+b[1]*tau**(1./3)+b[2]*tau+b[3]*tau**(3./2) #The brute force approach
            rho_star = x*b
            rho_star = rho_star.sum(axis=1) # To add up the rows (that pertain to a specific T_star)
        else:
            rho_star = np.zeros([len(tau)])
        return rho_star
    
    def rho_hat_2CLJQ(self,Temp,eps,sig,Lbond,Qpole,phase):
        M_w = self.M_w
        T_star = Temp/eps
        Q2_star = Qpole**2/(eps*sig**5)
        L_star = Lbond/sig
        rho_star = self.rho_star_hat_2CLJQ(T_star,Q2_star,L_star,phase)
        rho = rho_star *  M_w  / sig**3 / N_A * m3_to_nm3 * gm_to_kg #[kg/m3]
        return rho
    
    def rhol_hat_2CLJQ(self,Temp,eps,sig,Lbond,Qpole):
        rhol = self.rho_hat_2CLJQ(Temp,eps,sig,Lbond,Qpole,'liquid')
        return rhol #[kg/m3]
        
    def rhov_hat_2CLJQ(self,Temp,eps,sig,Lbond,Qpole):
        rhov = self.rho_hat_2CLJQ(Temp,eps,sig,Lbond,Qpole,'vapor')
        return rhov #[kg/m3]
    
    def Psat_star_hat_2CLJQ(self,T_star, q,l):
        b_c1, b_c2, b_c3 = self.b_c1, self.b_c2, self.b_c3
        x_c1 = [1., q**2, q**3, l**2/(l**2+0.75), l**3/(l**3+0.75), l**2*q**2/(l**2+0.75), l**3*q**2/(l**3+0.75), l**2*q**3/(l**2+0.75), l**3*q**3/(l**3+0.75)]
        x_c2 = [1., q**2, q**3, l**2/(l+0.75)**2, l**3/(l+0.75)**3, l**2*q**2/(l+0.75)**2, l**3*q**2/(l+0.75)**3, l**2*q**3/(l+0.75)**2, l**3*q**3/(l+0.75)**3]
        x_c3 = [q**2, q**5, l**0.5]
        c1 = (x_c1*b_c1).sum()
        c2 = (x_c2*b_c2).sum()
        c3 = (x_c3*b_c3).sum()
        Psat_star = np.exp(c1 + c2/T_star + c3/(T_star**4))
        return Psat_star
    
    def Psat_hat_2CLJQ(self,Temp,eps,sig,Lbond,Qpole):
        T_star = Temp/eps
        Q2_star = Qpole**2/(eps*sig**5)
        L_star = Lbond/sig
        Psat_star = self.Psat_star_hat_2CLJQ(T_star,Q2_star,L_star)
        Psat = Psat_star *  eps  / sig**3 * k_B * m3_to_nm3 * J_per_m3_to_kPA #[kPa]
        return Psat
    
    def LJ_model(self,r,eps,sig):
        r_star = r/sig
        U = 4 * eps * (r_star**(-12) - r_star**(-6))
        return U