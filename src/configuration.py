#!/usr/bin/env python
# coding: utf-8

import numpy as np


## Physical constants
g     = 9.81   # (m/s2)  Gravity constant

## Storms
Dmean = np.log(51.3)/2 # (-) Mean of lognormal distribution
Dsigma = np.log(1.4)   # (-) Sigma of lognormal distribution

## Water 
fetch = 30000  # (m)     Fetch of wind 
rho   = 1025   # (kg/m3) Density of water
h0    = 5      # (m)     Depth below mean sea level at start of lifetime  
h50   = 0.4    # (m)     Projected sea level rise in 50 years
h100  = 1      # (m)     Projected sea level rise in 100 years
hlake = 4      # (m)     Depth on lake side (assumed constant for now)

## Structural properties
hgate = 7.5      # (m)     Gate height (also z-coord of overhang)
width = 12       # (m)     Gate width
Ly    = 0.1      # (m)     Overhang length
cr    = 0.8      # (-)     Reflection coefficient

## Wave impact duration triangular distribution properties
tri_min  = 0.01  # (s)   Lower bound
tri_mode = 0.105 # (s)   Mode
tri_max  = 0.2   # (s)   Upper Bound

## Air cavitation normal distribution properties
beta_mean = 1.17 # (-)   Mean
beta_std  = 0.11 # (-)   Std

## Modeling settings
C_ts = 20        # (-)   Ratio between Tp and ts
dz = 0.05        # (m)   Vertical gate coordinate step size
dx = 0.05        # (m)   Horizontal gate coordinate step size
dt = 0.005       # (m)   Timestep for pressure time series
Dsegment = 1     # (h)   Duration of a single load situation (Along with ts also influences spectral resolution!)
hlife = 50       # (y)   Desired lifetime of structure

## JONSWAP
gamma = 3.3      # (-)   Amplification factor