'''Defines constants and settings used throughout model'''
import numpy as np
import os

## Physical constants
g     = 9.81   # (m/s2)  Gravity constant

## Water
F     = 30000  # (m)     Fetch of wind
rho   = 1025   # (kg/m3) Density of water
h0    = 5      # (m)     Depth below mean sea level at start of lifetime
H_LAKE= 4      # (m)     Depth on lake side (assumed constant)
cp    = 1500   # (m/s)   Speed of sound in water

## Structural properties
cr     = 0.8      # (-)     Reflection coefficient
rho_s  = 7850     # (kg/m3) Steel density

## Fatigue
gamma_Mf = 1.2
gamma_Ff = 1

## Wave impact duration triangular distribution properties
tri_min  = 0.01   # (s)   Lower bound
tri_mode = 0.105  # (s)   Mode
tri_max  = 0.2    # (s)   Upper Bound

## Air cavitation normal distribution properties
beta_mean = 1.17 # (-)   Mean
beta_std  = 0.11 # (-)   Std

## Modeling settings
n_scia_modes = 16# (-)   Amount of modes SCIA outputs (Not the same as # modes used in model!)
n_freqs = 4001   # (-)   Amount of frequencies to calculate FRF for
f_max = 200      # (Hz)  Maximum frequency to calculate FRF for
n_x = 4          # (-)   Amount of FSI-segments in x-direction
n_z = 5          # (-)   Amount of FSI-segments in z-direction
C_ts = 30        # (-)   Ratio between Tp and ts
dz = 0.05        # (m)   Vertical gate coordinate step size
dx = 0.05        # (m)   Horizontal gate coordinate step size
dt = 0.0025      # (s)   Timestep for pressure time series
N_HOURS = 1      # (h)   Duration of a single load situation
hlife = 80       # (y)   Desired lifetime of structure
t = np.arange(0,N_HOURS*3600+1*dt,dt)

## Coordinates
# x_coords = np.linspace(0, WIDTH, int(WIDTH/dx+1))
# z_coords = np.linspace(0, H_GATE, int(H_GATE/dz+1))
# t = np.arange(0,N_HOURS*3600+1*dt,dt)

# ## Gate properties
# Gate = None
# filepath = '../data/06_transferfunctions/currentcase.cp.pkl'
# class _Config:
#     def __init__(self):
#         if os.path.isfile(filepath):
#             with open(filepath, 'rb') as file:
#                 properties = cloudpickle.load(file)
#                 for key,val in properties.items():
#                     setattr(self,key,val)
#         else:
#             print('No system properties found. Run initialize() to create new file.')
         
#     def __getattr__(self, name):
#         try:
#             return self.config[name]
#         except KeyError:
#             return print('fail')

# Gate = _Config()
