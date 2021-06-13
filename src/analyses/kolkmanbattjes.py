import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pyfftw
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline

pyfftw.config.NUM_THREADS = 4
root_dir = os.path.join(os.getcwd(), '..')
sys.path.append(root_dir)

from src.configuration import N_HOURS, WIDTH, dz, H_GATE, z_coords, t, Gate, dt, H_LAKE, rho, cr, beta_mean
from src.utilities.nearest import nearest
from src.woodandperegrine import woodandperegrine
from src.stress import imp_discretize

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Helvetica'] + plt.rcParams['font.serif']
params = {'legend.fontsize': 'large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'large',
         'axes.titlesize':'large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
plt.rcParams.update(params)
TUred = "#c3312f"
TUblue = "#00A6D6"
TUgreen = "#00a390"

# Prepare Wood & Peregrine dimensionless impact shape
Pz_impact_U1 = woodandperegrine()

# Load FRF corresponding to loaded system properties
directory = '../data/06_transferfunctions/'+str(Gate.case)+'/FRF_'+str(Gate.n_modes)+'modes'+str(Gate.case)+'.npy'
try:
    frf_intpl = np.load(directory, mmap_mode='r')
    # Map mode only loads parts of 3Gb matrix when needed instead of keeping it all in RAM.
except OSError:
    print("An exception occurred: no FRF found at "+directory)

def single_wave(h_wave, t_wave, tau, duration=30, responsetype='stress', coords=(5,1,7.5), plot=False, modes=False):
    h_wave *= (1+cr)
    t_pulse = np.arange(0, duration, dt)
    if h_wave < 10**-8:
        return t_pulse, np.zeros(len(t_pulse))
    omegaw = 2*np.pi/t_wave
    U = omegaw*(h_wave/2)
    beta = beta_mean

    pz_peak = beta*U/(0.5*tau)
    shift = 0
    points_t = [0, shift, tau/2+shift, tau+shift, duration]
    points_p = [0, 0, 1, 0, 0]
    pulse = np.interp(t_pulse, points_t, points_p)*pz_peak
    coords = nearest(*coords)

    # Prepare Wood & Peregrine dimensionless impact shape
    Pz_impact_U1 = woodandperegrine()
    # Construct force-time matrix for all z-coordinates
    p_matrix = np.zeros((len(pulse), len(Pz_impact_U1)))
    np.multiply.outer(pulse, Pz_impact_U1, out = p_matrix)

    # does not integrate over x!
    p_sections = np.array([np.sum(np.split(p_matrix,Gate.ii_z[1:-1], axis=1)[i],axis=1)*dz                           /(z_coords[Gate.ii_z[i+1]]-z_coords[Gate.ii_z[i]]) for i in range(Gate.n_z)])

    # Find force spectrum for sections
    p_imp_f = pyfftw.interfaces.numpy_fft.rfft(p_sections,axis=1)
    f = np.fft.rfftfreq(len(pulse), d=dt)
    # Interpolate to correct resolution
    FRF_int = np.zeros((Gate.n_x, Gate.n_z, Gate.n_modes, len(f)), dtype='complex')
    for i in range(Gate.n_x):
        for j in range(Gate.n_z):
            for k in range(Gate.n_modes):
                func = interp1d(Gate.FRF_f,Gate.FRF[i][j][k],
                                bounds_error=False,fill_value=np.nan)
                FRF_int[i,j,k,:] = func(f)
    
    # Compute response for modes
    response_modes = np.zeros((Gate.n_modes, len(f)), dtype=np.complex)
    if responsetype == 'stress':
        if sum(Gate.stressneg3D.loc[coords] - Gate.stresspos3D.loc[coords])>0:
            mode_shape = [[i] for i in Gate.stressneg3D.loc[coords].to_list()]
        else:
            mode_shape = [[i] for i in Gate.stresspos3D.loc[coords].to_list()]
    elif responsetype == 'disp':
        mode_shape = [[-i] for i in Gate.disp3D.loc[coords].to_list()] # minus sign so direction matches forces
    else:
        print('Wrong response type provided')
        return

    np.einsum('jl,ijkl->kl', p_imp_f, FRF_int,
                  out = response_modes, dtype=np.complex)
#     np.multiply(np.multiply(p_imp_f[np.newaxis,:,np.newaxis,:],FRF_int).sum(axis=(0,1)),
#                 mode_shape, out = response_modes)
    response_tot = mode_shape*pyfftw.interfaces.numpy_fft.irfft(np.array([mode[~np.isnan(mode)] for mode in
                                                               response_modes]))
    if plot == True:
        if responsetype == 'stress':
            scale = 10**-6
            ylabel = 'Stress [MPa]'
        elif responsetype == 'disp':
            scale = 1000
            ylabel = 'Deflection [mm]'
        plotrange=[-.05,duration]
        fig, ax1 = plt.subplots(figsize=[15,5])
        ax1.set_xlim(plotrange[0], plotrange[1])
        ax1.set_xlabel('t [s]')
        section = np.where((t_pulse>plotrange[0]) & (t_pulse<plotrange[1]))[0]
        resp = ax1.plot(t_pulse[section],scale*response_tot.sum(axis=0)[section],
                       color=TUblue) 
        ax1.set_ylabel(ylabel, fontsize = 12)
        d_lim = 1.2*max(response_tot.sum(axis=0)[section])*scale
        d_mean = np.mean(response_tot.sum(axis=0)[section])*scale
        ax1.set_ylim(d_mean-d_lim, d_lim)
        ax1.grid(lw=0.25)
    if modes:
        return t_pulse, response_tot
    elif modes == False:
        return t_pulse, response_tot.sum(axis=0)

def effect_added_mass(m_gate):
    h_sea = H_GATE # Always the case during an impact
    h1d = max(1,h_sea/H_GATE)
    h2d = max(1,H_LAKE/H_GATE)
#     cr_damp = 0.01 # From sheet provided by marco, add source
    CL = 0.583 # Westergaard. From Kolkman graph, always this value for this type of system
    m_water_sea = CL*rho*min(h_sea,H_GATE)**2*WIDTH
    m_water_lake = CL*rho*min(H_LAKE,H_GATE)**2*WIDTH
    m_water = m_water_sea+m_water_lake
    m_tot = m_water+m_gate

    fn_droog = Gate.eigenfreqs[0] # Probably not accurate due to presence of other modes
    fn_nat = fn_droog/np.sqrt(m_tot/m_gate)
    omega_nat = 2*np.pi*fn_nat

#     psi1 = omega_nat**2*h_sea/9.81
#     psi2 = omega_nat**2*H_LAKE/9.81
#     added_damp_sea = cr_damp*rho*omega_nat*min(H_GATE,h_sea)**2*WIDTH 
#     added_damp_lake = cr_damp*rho*omega_nat*min(H_GATE,H_LAKE)**2*WIDTH
#     added_damp = added_damp_sea+added_damp_lake
#     damp_crit = 2*m_tot*omega_nat
#     damp_ratio_sea = added_damp_sea/damp_crit
#     damp_ratio_lake = added_damp_lake/damp_crit
#     damp_ratio = damp_ratio_sea+damp_ratio_lake
#     total_damp_ratio = 0.01+damp_ratio #Rule of thumb
    return m_tot, omega_nat

# def dynamic_amplification(m_gate):
#     m_tot, omega_nat = effect_added_mass(m_gate)
#     eta = .02 #make variable
#     w = Gate.f_trunc*2*np.pi
#     daf = np.sqrt((1-w**2/omega_nat**2)**2+(2*eta*w/omega_nat)**2)**-1
#     return daf

# def single_wave_daf(h_wave, t_wave, m_gate, coords=(5,1,7.5), tau=0.1, responsetype='stress', static=False):
#     coords = nearest(*coords)
#     if responsetype == 'stress': ##CHANGE!
#         if sum(Gate.stressneg3D.loc[coords] - Gate.stresspos3D.loc[coords])>0:
#             mode_shape = [[i] for i in Gate.stressneg3D.loc[coords].to_list()]
#         else:
#             mode_shape = [[i] for i in Gate.stresspos3D.loc[coords].to_list()]
#     elif responsetype == 'disp':
#         mode_shape = [[-i] for i in Gate.disp3D.loc[coords].to_list()] # minus sign so direction matches forces
#     else:
#         print('Wrong response type provided')
#         return

#     omegaw = 2*np.pi/t_wave
#     U = (1+cr)*omegaw*(h_wave/2)
#     beta = beta_mean
    
#     pz_peak = beta*U/(0.5*tau)
#     t_pulse = np.arange(0,tau,dt)
#     shift = 0
#     points_t = [0, shift, tau/2+shift, tau+shift, 3600]
#     points_p = [0, 0, 1, 0, 0]
#     pulse = np.interp(t, points_t, points_p)*pz_peak
#     p_sections = imp_discretize(pulse)
#     p_imp_f = pyfftw.interfaces.numpy_fft.rfft(p_sections,axis=1)

#     if static:
#         response_imp = np.zeros(len(Gate.f_trunc), dtype=np.complex)
#         frf_static = Gate.FRF[:,:,:,0] # Zero frequency
#         np.einsum('jl,ijk->l', p_imp_f, frf_static,
#                   out = response_imp, dtype=np.complex)
#         response_imp_t = mode_shape[0]*pyfftw.interfaces.numpy_fft.irfft(response_imp)
#         response_imp_t = np.append(response_imp_t,response_imp_t[-1])
#         return response_imp_t    
#     else:
#         # Mode shape and FRF
#         daf = dynamic_amplification(m_gate)
#         frf_static = np.multiply.outer(Gate.FRF[:,:,:,0], daf) # Zero frequency

#         ## Compute response for modes
#         response_imp = np.zeros(len(Gate.f_trunc), dtype=np.complex)
#         np.einsum('jl,ijkl->l', p_imp_f, frf_static,
#                   out = response_imp, dtype=np.complex)
#         response_imp_t = mode_shape[0]*pyfftw.interfaces.numpy_fft.irfft(response_imp)
#         response_imp_t = np.append(response_imp_t,response_imp_t[-1])
#         return response_imp_t    
    
def single_wave_U(U, tau, duration=30, responsetype='stress', coords=(5,1,7.5), plot=False, modes=False):
    beta = beta_mean
    t_pulse = np.arange(0, duration, dt)
    pz_peak = beta*U/(0.5*tau)
    shift = 0
    points_t = [0, shift, tau/2+shift, tau+shift, duration]
    points_p = [0, 0, 1, 0, 0]
    pulse = np.interp(t_pulse, points_t, points_p)*pz_peak
    coords = nearest(*coords)

    # Prepare Wood & Peregrine dimensionless impact shape
    Pz_impact_U1 = woodandperegrine()
    # Construct force-time matrix for all z-coordinates
    p_matrix = np.zeros((len(pulse), len(Pz_impact_U1)))
    np.multiply.outer(pulse, Pz_impact_U1, out = p_matrix)

    # does not integrate over x!
    p_sections = np.array([np.sum(np.split(p_matrix,Gate.ii_z[1:-1], axis=1)[i],axis=1)*dz                           /(z_coords[Gate.ii_z[i+1]]-z_coords[Gate.ii_z[i]]) for i in range(Gate.n_z)])

    # Find force spectrum for sections
    p_imp_f = pyfftw.interfaces.numpy_fft.rfft(p_sections,axis=1)
    f = np.fft.rfftfreq(len(pulse), d=dt)
    # Interpolate to correct resolution
    FRF_int = np.zeros((Gate.n_x, Gate.n_z, Gate.n_modes, len(f)), dtype='complex')
    for i in range(Gate.n_x):
        for j in range(Gate.n_z):
            for k in range(Gate.n_modes):
                func = interp1d(Gate.FRF_f,Gate.FRF[i][j][k],
                                bounds_error=False,fill_value=np.nan)
                FRF_int[i,j,k,:] = func(f)
    
    # Compute response for modes
    response_modes = np.zeros((Gate.n_modes, len(f)), dtype=np.complex)
    if responsetype == 'stress':
        if sum(Gate.stressneg3D.loc[coords] - Gate.stresspos3D.loc[coords])>0:
            mode_shape = [[i] for i in Gate.stressneg3D.loc[coords].to_list()]
        else:
            mode_shape = [[i] for i in Gate.stresspos3D.loc[coords].to_list()]
    elif responsetype == 'disp':
        mode_shape = [[-i] for i in Gate.disp3D.loc[coords].to_list()] # minus sign so direction matches forces
    else:
        print('Wrong response type provided')
        return

    np.einsum('jl,ijkl->kl', p_imp_f, FRF_int,
                  out = response_modes, dtype=np.complex)
#     np.multiply(np.multiply(p_imp_f[np.newaxis,:,np.newaxis,:],FRF_int).sum(axis=(0,1)),
#                 mode_shape, out = response_modes)
    response_tot = mode_shape*pyfftw.interfaces.numpy_fft.irfft(np.array([mode[~np.isnan(mode)] for mode in
                                                               response_modes]))
    if plot == True:
        if responsetype == 'stress':
            scale = 10**-6
            ylabel = 'Stress [MPa]'
        elif responsetype == 'disp':
            scale = 1000
            ylabel = 'Deflection [mm]'
        plotrange=[-.05,duration]
        fig, ax1 = plt.subplots(figsize=[15,5])
        ax1.set_xlim(plotrange[0], plotrange[1])
        ax1.set_xlabel('t [s]')
        section = np.where((t_pulse>plotrange[0]) & (t_pulse<plotrange[1]))[0]
        resp = ax1.plot(t_pulse[section],scale*response_tot.sum(axis=0)[section],
                       color=TUblue) 
        ax1.set_ylabel(ylabel, fontsize = 12)
        d_lim = 1.2*max(response_tot.sum(axis=0)[section])*scale
        d_mean = np.mean(response_tot.sum(axis=0)[section])*scale
        ax1.set_ylim(d_mean-d_lim, d_lim)
        ax1.grid(lw=0.25)
    if modes:
        return t_pulse, response_tot
    elif modes == False:
        return t_pulse, response_tot.sum(axis=0)    
