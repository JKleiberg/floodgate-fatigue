#!/usr/bin/env python
# coding: utf-8

## Comments from original Matlab file

import numpy as np
def woodandperegrine(a,dz):
    from src.configuration import width, rho
    
    x = np.transpose([np.linspace(0,width,int(width/0.005+1))]*(round((1/dz)*a)+1))
    y = (200*width+1)*[np.linspace(0,a,int(a/dz+1))]
    z = x + np.multiply(1j,y)

    
    ## STEP 1 - CONFORMAL MAP 
    w= np.cosh(np.multiply(np.pi/a,z))
    
    ## STEP 2 - TRANSLATION AND MAGNIFICATION
    M = 2/(np.cosh(np.pi/a)-1)
    N = M + 1;
    h = M*w + N; 
    
    ## STEP 3 - CONFORMAL MAP 2
    zeta = a*np.arccosh(h)/np.pi; 
    ksi = np.real(zeta);
    eta = np.imag(zeta);
    
    ## set variables for integration over eta
    Nint     = 1000;
    deta_int = a/Nint;
    eta_int  = np.multiply([x - 1/2 for x in range(1,1001)],deta_int);
    b_int    = (np.cos(np.multiply(np.pi/a,eta_int))-N)/M ;
    
    P = 0
    
    for n in range(1,31):
        alphan = (n-1 + 1/2)*np.pi/a; # I sum from n=0 , so take n-1 such that n can be used as index
        # numerical integration to determine Fourier coefficients
        An = 2/alphan/a * sum(np.sin(np.pi*eta_int/a)*np.cos(alphan*eta_int)/np.sqrt(b_int**2-1) / M ) * deta_int  # says Am in paper, I don't see where a comes from
        # % sum Fourier modes to get P
        P += An * np.cos(alphan*eta) * np.exp(-alphan*ksi) # not sure about the real() in right expression in paper
    return P[1,:]*rho   #per m width

