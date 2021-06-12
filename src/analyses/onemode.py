""" Calculates the stress at a specified point in the gate for 1 mode,
based on previously calculated quasi-static stress spectra and wave impact time series.

"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pyfftw
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from src.configuration import N_HOURS, WIDTH, dz, H_GATE, z_coords, t, Gate, dt, H_LAKE, rho, cr, beta_mean
from src.utilities.nearest import nearest
from src.woodandperegrine import woodandperegrine
from src.stress import imp_discretize
pyfftw.config.NUM_THREADS = 4
root_dir = os.path.join(os.getcwd(), '..')
sys.path.append(root_dir)

# Prepare Wood & Peregrine dimensionless impact shape
Pz_impact_U1 = woodandperegrine()

# Load FRF corresponding to loaded system properties
directory = '../data/06_transferfunctions/'+str(Gate.case)+'_1mode/FRF_1modes'+str(Gate.case)+'_1mode.npy'
try:
    frf_intpl = np.load(directory, mmap_mode='r')
    # Map mode only loads parts of 3Gb matrix when needed instead of keeping it all in RAM.
except OSError:
    print("An exception occurred: no FRF found at "+directory)
func_small = interp1d(Gate.f_trunc, frf_intpl, axis=3, fill_value= 'extrapolate')
frf_small = func_small(Gate.FRF_f)

def plot_stress_time(F_tot, response_t, coords, t_range):
    """Plots result of stress_time"""
    section = np.where((t>t_range[0]) & (t<t_range[1]))[0]
    fig, ax1 = plt.subplots(figsize=[15,5])
    ax2 = ax1.twinx()
#     ax1.set_title('Load and response at '+str(coords),fontsize = 14)
    ax1.set_xlim(t_range)
    ax1.set_xlabel('t [s]')
    resp = ax1.plot(t[section], response_t[section]/10**6, color="#00A6D6",
                    label='Equivalent gate stress')
    ax1.set_ylabel('Stress [MPa]', fontsize=12)
    d_max = 1.2*max(response_t[section])/10**6
    d_mean = np.mean(response_t[section])/10**6
    ax1.set_ylim(d_mean-d_max,d_max)
    ax1.legend()

    force = ax2.plot(t[section],F_tot[section]/1000, color="#c3312f", label = 'Wave force')
    ax2.set_ylabel('Integrated wave force [$kN/m$]', fontsize = 12)
    F_lim = 1.2*max(F_tot[section])/1000
    F_mean = np.mean(F_tot[section]/1000)
    ax2.set_ylim(F_mean-F_lim,F_lim)

    lines = resp + force
    labs = [l.get_label() for l in lines]
    ax1.grid(lw=0.25)
    ax1.legend(lines,labs, fontsize = 12)
    return fig

# Stress functions
def qs_discretize(pqs_f):
    """Integrates the quasi-static pressure spectrum over the gate sections."""
    # does not integrate over x!
    # Becomes problem if segments are variable or inputs not uniform in x-direction
    return np.array([np.sum(np.split(pqs_f,Gate.ii_z[1:-1], axis=0)[i],axis=0)*dz/\
                     (z_coords[Gate.ii_z[i+1]]-z_coords[Gate.ii_z[i]])\
                     for i in range(Gate.n_z)])

def imp_discretize(impact_force_t):
    """Integrates the impulsive pressure time series over the gate sections."""
    # Construct force-time matrix for all z-coordinates
    p_matrix = np.zeros((len(impact_force_t), len(Pz_impact_U1)))
    np.multiply.outer(impact_force_t, Pz_impact_U1, out = p_matrix)
    # does not integrate over x!
    # Becomes problem if segments are variable or inputs not uniform in x-direction
    return np.array([np.sum(np.split(p_matrix,Gate.ii_z[1:-1], axis=1)[i],axis=1)*dz/\
                     (z_coords[Gate.ii_z[i+1]]-z_coords[Gate.ii_z[i]]) for i in range(Gate.n_z)])

def stress_time_1mode(freqs, pqs_f, impact_force_t, coords, stresstype = None, 
                plot=False, plotrange = [50,80]):
    """Generates a time series of the stress at a given gate coordinate.

    Parameters:
    freqs: Frequencies of JONSWAP spectrum (Hz)
    pqs_f:  Quasi-static wave pressure spectrum (N/m2)
    impact_force_t: Wave impact time series (s)
    coords: X,Y,Z gate coordinates (m)
    plot: Whether to plot the result (True/False)

    Returns:

    F_tot: Wave force integrated over gate height (N/m)
    response_tot_t: Time series of gate stress at coordinate (N/m2)

    """

    # IRFFT of quasi-static part
    pqs_sections = qs_discretize(pqs_f)
    q_tot = pqs_sections.sum(axis=(0))*(H_GATE/Gate.n_z)
    # N/m (only works if sections are of constant size)
    # Interpolate FRF to correct resolution
    func = interp1d(Gate.FRF_f, frf_small, axis=3)
    frf_qs = func(freqs)
    if stresstype == 'pos':
        mode_shape = Gate.stresspos3D.loc[coords].to_list()
    elif stresstype == 'neg':
        mode_shape = Gate.stressneg3D.loc[coords].to_list()
    else:
        if sum(Gate.stressneg3D.loc[coords] - Gate.stresspos3D.loc[coords])>0:
            mode_shape = Gate.stressneg3D.loc[coords].to_list()
        else:
            mode_shape = Gate.stresspos3D.loc[coords].to_list()
    response_qs = np.zeros(len(freqs), dtype=np.complex)
    np.einsum('jl,ijkl->l', pqs_sections, frf_qs,
              out=response_qs, dtype=np.complex)
    response_qs_t = mode_shape[0]*len(response_qs)*pyfftw.interfaces.numpy_fft.irfft(response_qs)
    t_qs = np.linspace(0, 3600*N_HOURS, 2*(len(pqs_f[0])-1))
    # Interpolate qs
    func_qs = InterpolatedUnivariateSpline(t_qs, response_qs_t)
    resp_qs_intpl = func_qs(t)

    F_qs = pyfftw.interfaces.numpy_fft.irfft(q_tot)*len(q_tot)
    func_F_qs = InterpolatedUnivariateSpline(t_qs, F_qs)
    F_qs_intpl = func_F_qs(t)

    # IRFFT of impulsive part
    p_sections = imp_discretize(impact_force_t)
    p_imp_f = pyfftw.interfaces.numpy_fft.rfft(p_sections, axis=1)
    # Find total force
    F_gate = p_sections.sum(axis=(0))*(H_GATE/Gate.n_z)
    # N/m over width (only works if sections are of constant size)

    # Compute response for modes
    response_imp = np.zeros(len(Gate.f_trunc), dtype=np.complex)
    np.einsum('jl,ijkl->l', p_imp_f, frf_intpl,
              out=response_imp, dtype=np.complex)
    response_imp_t = mode_shape[0]*pyfftw.interfaces.numpy_fft.irfft(response_imp)
    response_imp_t = np.append(response_imp_t,response_imp_t[-1])

    # Combine responses and forces
    response_tot_t = resp_qs_intpl + response_imp_t
    F_tot = F_qs_intpl + F_gate
    if plot:
        plot_stress_time(F_tot, response_tot_t, coords, plotrange)
    return F_tot, response_tot_t

def single_wave_1mode(h_wave, t_wave, tau, duration=30, responsetype='stress', coords=(5,1,7.5), plot=False):
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
    FRF_int = np.zeros((Gate.n_x, Gate.n_z, len(f)), dtype='complex')
    for i in range(Gate.n_x):
        for j in range(Gate.n_z):
            func = interp1d(Gate.FRF_f, frf_small[i][j],
                            bounds_error=False, fill_value=np.nan)
            FRF_int[i,j,:] = func(f)
    
    # Compute response for modes
    response_modes = np.zeros(len(f), dtype=np.complex)
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

    np.einsum('jl,ijl->l', p_imp_f, FRF_int,
                  out = response_modes, dtype=np.complex)
    response_tot = mode_shape[0]*pyfftw.interfaces.numpy_fft.irfft(np.array(response_modes[~np.isnan(response_modes)]))
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

    return t_pulse, response_tot