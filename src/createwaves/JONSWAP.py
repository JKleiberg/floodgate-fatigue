#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
from scipy.optimize import root
from scipy.fftpack import fft
from src.configuration import g, gamma, Dsegment

# In[12]:


def JONSWAP(Hs, Tp, hseg, ts):
    ## Define frequencies
    f0 = 1/(Dsegment*3600)
    Nd = int(Dsegment*3600/ts)
    fp = 1/Tp
    f = np.fft.rfftfreq(Nd, ts)
    N = len(f)
    omega = 2*np.pi*f 
    
    ## Determine k for each frequency. Made faster with Airy wave theory
        # Transitions from shallow to intermediate to deep water calculated below,
        # only have to do costly solver for intermediate values.
    def wavenumber(x,oi):
        return oi**2 - x*g*np.tanh(x*hseg)
    
    shallow = np.sqrt(2*np.pi*g/(20*hseg)*np.tanh(2*np.pi/20))/(2*np.pi)
    deep = np.sqrt(np.pi*g/(hseg)*np.tanh(np.pi))/(2*np.pi)   
    
    k = []
    for i in range(len(omega)):
        if abs(f[i]) < shallow:
            k.append(abs(omega[i])/np.sqrt(g*hseg))
        elif abs(f[i]) > deep:
            k.append(omega[i]**2/g)
        else:
            k.append(root(wavenumber, 0.01, args=(omega[i])).x[0])
        
    ## JONSWAP
    amp = np.zeros(N, dtype=np.complex)
    phase = [random.uniform(-np.pi,np.pi) for _ in range(N)] # Generate random phases
    
    for i in range(N):
        af = abs(f[i])
        if af == 0:
            continue
        if af > fp:
            sigma = 0.09
        else:
            sigma = 0.07
        alpha = -(af-fp)**2/(2*sigma**2*fp**2)         
        S = 0.3125*(1-0.287*np.log(gamma))*Hs**2*fp**5*af**-5*np.exp(-5/4*(fp/af)**4)*gamma**np.exp(alpha)  # JONSWAP spectrum
        spec =  np.sqrt(2*S*f0)             # From power density to amplitude    
        amp[i]  = spec*np.exp(1j*phase[i])  # Add random phase
        
    return f,amp,k


# In[ ]:




