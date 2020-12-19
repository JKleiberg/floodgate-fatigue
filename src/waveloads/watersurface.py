#!/usr/bin/env python
# coding: utf-8

# In[46]:



import numpy as np

def watersurface(hseg,k,ts,amp,z):
    from src.configuration import cr, hgate, Dsegment, rho, g, hlake
    
    ## Apply coefficient for reflection of waves to amplitude spectrum
    amp_gate = (1+cr)*amp

    ## Surface elevation and zero crossings (crossing of overhang level)
    eta = np.fft.irfft(amp_gate)*2*(len(k)-1)                           # Generate water surface time domain from spectrum. Scaled by 2(k-1) because of IRFFT. 
   
    ## Setting ends of time series to zero to prevent discontinuities
    len_eta = len(eta)
    check = eta < 0
    first = (check[:-1] & ~check[1:]).nonzero()[0][1]
    last = (check[:-1] & ~check[1:]).nonzero()[0][-1]
    eta = eta[first+1:last]                                               # Truncate array at first and last zero crossings
    front = int((len_eta-len(eta))/2)
    eta = np.pad(eta, (front, int(len_eta-len(eta)-front)), 'constant')            # Zero pad truncated array back to old length

    ## Hydrostatic pressure
    fhs_z     = []                                                        # Hydrostatic pressure as function of z

    for i in range(len(z)):
        if (hgate-z[i]) <= hlake:
            fhs_L = rho*g*(-hgate + hlake + z[i])
        elif (hlake > hgate):
            fhs_L = rho*g*(z[i]+(hlake-hgate))  
        else:
            fhs_L = 0
        if (hgate-z[i]) <= hseg:
            fhs_S = rho*g*(-hgate + hseg + z[i])
        elif (hseg > hgate):
            fhs_S = rho*g*(z[i]+(hseg-hgate))  
        else:
            fhs_S = 0
        fhs_z.append(fhs_S-fhs_L)

    ## Quasi-static pressure     
    fqs_fz    = []
    
    # Split spectrum based on whether it is deep, shallow, or intermediate water depth
    shallow_k = np.pi/(10*hseg)
    deep_k = np.pi/hseg
    k = np.array(k)
    # Define precise function, and approximations for deep and shallow water
    func_shallow = lambda a, k, zc: a*rho*g
    func_shallow_res = np.frompyfunc(func_shallow, 3,1)
    func_inter = lambda a, k, zc: a*(rho*g*np.cosh(k*(hgate-zc))/np.cosh(k*hseg))
    func_inter_res = np.frompyfunc(func_inter, 3,1)
    func_deep = lambda a, k, zc: a*rho*g*np.exp(k*(-zc))
    func_deep_res = np.frompyfunc(func_deep, 3,1)
    
    for j in range(len(z)):       #Check difference between depth and gate height!
        fqs_z =[]
        if hseg >= (hgate-z[j]): 
            p_shallow = func_shallow_res(amp_gate[k<shallow_k],k[k<shallow_k],z[j])
            p_inter = func_inter_res(amp_gate[(k>=shallow_k) & (k<=deep_k)],k[(k>=shallow_k) & (k<=deep_k)],z[j])
            p_deep = func_deep_res(amp_gate[k>deep_k],k[k>deep_k],z[j])
            fqs_z = np.concatenate([p_shallow,p_inter,p_deep])
        else:
            fqs_z = np.zeros(len(k))
        fqs_z[0] = fhs_z[j]                # Add static load at zero frequency
        fqs_fz.append(fqs_z)        
        
    return eta, np.array(fqs_fz)      

