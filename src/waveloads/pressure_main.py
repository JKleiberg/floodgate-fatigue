#!/usr/bin/env python
# coding: utf-8

# In[7]:


from src.configuration import dt
import pandas as pd

def interpolate(eta,t):
    p = pd.DataFrame(eta,columns=['eta'])
    p.index=pd.to_timedelta(t,unit='s') 
    pi = p.resample(str(dt)+'S').mean()

    ind = pi.index
    pi.index = pi.index.total_seconds()
    pi = pi.interpolate()#method='index')
    pi.index = ind
    return pi

import numpy as np
from src.configuration import hgate, tri_min, tri_mode, tri_max, beta_mean, beta_std

def impactloads(pi,Imp_only):
    ## Determine impact velocities
    pos = np.array(pi['eta'] < hgate)
    crossings = (pos[:-1] & ~pos[1:]).nonzero()[0]+1  # Find when water surface 'hits' overhang
    U_t = pi['eta'].diff() / pi['eta'].index.to_series().diff().dt.total_seconds()          # Velocity at every t  
    U_cross = U_t[crossings].to_numpy()                         # Velocities at zero-crossings
    pi.columns = ['F']
    pi['F'] = 0
    if Imp_only == True:
        return U_cross
    ## Probabilistic impact properties
    tau = np.random.triangular(tri_min,tri_mode,tri_max, size = len(U_cross)) ## Generate random tau from triangular dist.
    beta = np.random.normal(beta_mean,beta_std, size = len(U_cross))   ## Generate random normally distributed beta
    U_cross = U_cross/(tau*0.5)*beta      ## Impulse to force

    for i in range(len(crossings)):
        steps = np.int(np.ceil(tau[i]/dt/2)*2+1)
        up = np.linspace(0,U_cross[i],int((steps+1)/2))
        down = np.linspace(U_cross[i],0,int((steps+1)/2))
        pulse = np.concatenate([up,down[1:]])
        if (crossings[i]+steps) < len(pi['F']):
            pi['F'].iloc[crossings[i]:int(crossings[i]+steps)] += pulse
   
    return pi['F'].to_numpy()

#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from src.configuration import rho, g, Ly, cr, dz, Dsegment,C_ts
from src.createwaves.JONSWAP import JONSWAP
from src.waveloads.watersurface import watersurface
from src.waveloads.woodandperegrine import woodandperegrine

## Wood and Perigrine pressure impulse shape
z = np.linspace(0, hgate, int(hgate/dz+1))

def pressure(Hs, Tp, hsea, Imp_only = False):
    
    H_SEA = round(hsea,1)
    ts = round(Tp/C_ts,3)

    ## Generate JONSWAP spectrum 
    f, amp, k = JONSWAP(Hs, Tp, H_SEA,ts)

    ## Create water surface and linear wave pressures from spectrum in time domain with duration D
    eta, pqs_f = watersurface(H_SEA,k,ts,amp,z)

    t = np.linspace(0,3600*Dsegment,len(eta))             # Total time vector

    wl_tot = eta + H_SEA  # Add depth to wave fluctuations
    
    if (wl_tot < hgate).all():
        if Imp_only:
            return 0
        else:
            return pqs_f, np.zeros(int(3600*Dsegment/dt+1))
    
    wl_int = interpolate(wl_tot,t)
    p_imp_t = impactloads(wl_int,Imp_only)
    
    if Imp_only==True:
        wp_a = hgate/Ly
        wp_dz = dz/Ly

        Pz_impact_U1 = Ly*woodandperegrine(wp_a,wp_dz) # Dimensionless impact shape
        Pz_impact_U1 = Pz_impact_U1[::-1]  #Reverse it so the order matches the coordinates of the other pressure matrices
        imp_peak = pd.DataFrame(np.multiply.outer(p_imp_t,Pz_impact_U1)).T
        if imp_peak.empty:
            imp_tot = 0
        else:
            imp_z = imp_peak.iloc[:, :].apply(lambda x: np.trapz(x,z)) ## Integrate impulses over depth
            imp_tot = int(sum(imp_z)) ## Sum impulses
        return imp_tot
    
    return pqs_f, p_imp_t

