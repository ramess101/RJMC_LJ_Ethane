import numpy as np
import math
import yaml
import scipy as sp

# Conversion constants

k_B = 1.38065e-23 #[J/K]
N_A = 6.02214e23 #[1/mol]
m3_to_nm3 = 1e27
gm_to_kg = 1./1000
J_to_kJ = 1./1000
J_per_m3_to_kPA = 1./1000

class LennardJones():
    def __init__(self,M_w):
        
        self.M_w = M_w
        
        with open("LJ_fluid.yaml") as yfile:
            yfile = yaml.load(yfile)
    
        self.T_c_star = yfile["correlation_parameters"]["Lofti"]["T_c_star"]
        self.rhoc_star = yfile["correlation_parameters"]["Lofti"]["rho_c_star"]
        self.rhol_star_params = yfile["correlation_parameters"]["Lofti"]["rho_L_star_params"]
        self.rhov_star_params = yfile["correlation_parameters"]["Lofti"]["rho_v_star_params"]
        self.Psat_star_params = yfile["correlation_parameters"]["Lofti"]["P_v_star_params"]
        self.deltaHv_star_params = yfile["correlation_parameters"]["Lofti"]["HVP_star_params"]
        
        self.rhol_star_params = [self.rhoc_star, self.T_c_star, self.rhol_star_params[0], self.rhol_star_params[1], self.rhol_star_params[2]]
        self.rhov_star_params = [self.rhoc_star, self.T_c_star, self.rhov_star_params[0], self.rhov_star_params[1], self.rhov_star_params[2]]
        
    def rhol_star_hat(self,T_star):
        b = self.rhol_star_params
        tau = np.ones(len(T_star))*b[1] - T_star # T_c_star - T_star
        rhol_star = b[0] + b[2]*tau**(1./3) + b[3]*tau + b[4]*tau**(3./2)
        return rhol_star
    
    def rhol_hat_LJ(self,T,eps,sig):
        M_w = self.M_w
        T_star = T/(np.ones(len(T))*eps)
        rhol_star = self.rhol_star_hat(T_star)
        rhol = rhol_star *  M_w  / sig**3 / N_A * m3_to_nm3 * gm_to_kg #[kg/m3]
        return rhol
    
    def rhov_star_hat(self,T_star):
        b = self.rhov_star_params
        tau = np.ones(len(T_star))*b[1] - T_star # T_c_star - T_star
        rhov_star = b[0] + b[2]*tau**(1./3) + b[3]*tau + b[4]*tau**(3./2)
        return rhov_star
    
    def rhov_hat_LJ(self,T,eps,sig):
        M_w = self.M_w
        T_star = T/(np.ones(len(T))*eps)
        rhov_star = self.rhov_star_hat(T_star)
        rhov = rhov_star *  M_w  / sig**3 / N_A * m3_to_nm3 * gm_to_kg #[kg/m3]
        return rhov
    
    def Psat_star_hat(self,T_star):
        b = self.Psat_star_params
        Psat_star = np.exp(b[0]*T_star + b[1]/T_star + b[2]/(T_star**4))
        return Psat_star
    
    def Psat_hat_LJ(self,T,eps,sig):
        T_star = T/(np.ones(len(T))*eps)
        Psat_star = self.Psat_star_hat(T_star)
        Psat = Psat_star *  eps  / sig**3 * k_B * m3_to_nm3 * J_per_m3_to_kPA #[kPa]
        return Psat
    
    def deltaHv_star_hat(self,T_star):
        T_c, b = self.T_c_star, self.deltaHv_star_params
        tau = np.ones(len(T_star))*T_c - T_star # T_c_star - T_star
        deltaHv_star = b[0]*tau**(1./3) + b[1]*tau**(2./3) + b[2]*tau**(3./2)
        return deltaHv_star
    
    def deltaHv_hat_LJ(self,T,eps):
        T_star = T/(np.ones(len(T))*eps)
        deltaHv_star = self.deltaHv_star_hat(T_star)
        deltaHv = deltaHv_star * eps * k_B * N_A * J_to_kJ #[kJ/mol]
        return deltaHv
    
    def B2_hat_LJ(self,T,eps,sig):
        if eps == 0:
            pass
        T_star = T/(np.ones(len(T))*eps)
        B2_star = np.zeros(len(T))
        n = np.arange(0,31)
        for i,t_star in enumerate(T_star):
            addend = pow(2,(2*n+1.)/2)*pow(1./t_star,(2*n+1.)/4)*sp.special.gamma((2*n-1.)/4)/(4*sp.misc.factorial(n))
            #B2_star[i] = addend.sum() # This is the standard approach but sometimes 'nan' results
            B2_star[i] = np.nansum(addend) # This can handle even 'nan' results
        B2_star *= -2./3 * math.pi * sig**3
        B2 = B2_star * N_A / m3_to_nm3 #[m3/mol]
        return B2
    
    def LJ_model(self,r,eps,sig):
        r_star = r/sig
        U = 4 * eps * (r_star**(-12) - r_star**(-6))
        return U

    def calc_eps_Tc(self,Tc):
        eps_Tc = Tc/self.T_c_star 
        return eps_Tc #[K]
    
    def calc_sig_rhoc(self,rhoc):
        sig_rhoc = (self.rhoc_star / rhoc * self.M_w  / N_A * m3_to_nm3 * gm_to_kg)**(1./3) #[nm]
        return sig_rhoc