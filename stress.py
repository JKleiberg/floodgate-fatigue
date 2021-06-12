""" Calculates the stress at a specified point in the gate,
based on previously calculated quasi-static stress spectra and wave impact time series.

"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pyfftw
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from src.configuration import N_HOURS, WIDTH, dz, H_GATE, z_coords, t, Gate
from src.woodandperegrine import woodandperegrine
pyfftw.config.NUM_THREADS = 4
root_dir = os.path.join(os.getcwd(), '..')
sys.path.append(root_dir)

# Prepare Wood & Peregrine dimensionless impact shape
Pz_impact_U1 = woodandperegrine()

# Load FRF corresponding to loaded system properties
directory = '../data/06_transferfunctions/'+str(Gate.case)+'/FRF_'+str(Gate.n_modes)+'modes'+str(Gate.case)+'.npy'
try:
    frf_intpl = np.load(directory, mmap_mode='r')
    # Map mode only loads parts of 3Gb matrix when needed instead of keeping it all in RAM.
except OSError:
    print("An exception occurred: no FRF found at "+directory)

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

    force = ax2.plot(t[section], F_tot[section]/1000, color="#c3312f", label = 'Wave force')
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

def stress_time(freqs, pqs_f, impact_force_t, coords, stresstype = None, 
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
    func = interp1d(Gate.FRF_f, Gate.FRF, axis=3)
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
    np.einsum('jl,ijkl,k->l',pqs_sections, frf_qs, mode_shape,
              out=response_qs, dtype=np.complex)
    response_qs_t = len(response_qs)*pyfftw.interfaces.numpy_fft.irfft(response_qs)
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
    np.einsum('jl,ijkl,k->l', p_imp_f, frf_intpl, mode_shape,
              out=response_imp, dtype=np.complex)
    response_imp_t = pyfftw.interfaces.numpy_fft.irfft(response_imp)
    response_imp_t = np.append(response_imp_t,response_imp_t[-1])

    # Combine responses and forces
    response_tot_t = resp_qs_intpl + response_imp_t
    F_tot = F_qs_intpl + F_gate
    if plot:
        plot_stress_time(F_tot, -response_tot_t, coords, plotrange)
    return F_tot, response_tot_t

def stress_per_mode(freqs, pqs_f, impact_force_t, coords, stresstype = None):
    # IRFFT of quasi-static part
    pqs_sections = qs_discretize(pqs_f)
    # N/m (only works if sections are of constant size)
    # Interpolate FRF to correct resolution
    func = interp1d(Gate.FRF_f, Gate.FRF, axis=3)
    frf_qs = func(freqs)
    if   stresstype == 'pos':
        mode_shape = Gate.stresspos3D.loc[coords].to_list()
    elif stresstype == 'neg':
        mode_shape = Gate.stressneg3D.loc[coords].to_list()
    else:
        if sum(Gate.stressneg3D.loc[coords] - Gate.stresspos3D.loc[coords])>0:
            mode_shape = Gate.stressneg3D.loc[coords].to_list()
        else:
            mode_shape = Gate.stresspos3D.loc[coords].to_list()
    response_qs = np.zeros([Gate.n_modes,len(freqs)], dtype=np.complex)
    np.einsum('jl,ijkl,k->kl', pqs_sections, frf_qs, mode_shape,
              out = response_qs, dtype=np.complex)
    response_qs_t = len(response_qs)*pyfftw.interfaces.numpy_fft.irfft(response_qs)
    t_qs = np.linspace(0, 3600*N_HOURS, 2*(len(pqs_f[0])-1))
    # Interpolate qs

    resp_qs_intpl = np.zeros([Gate.n_modes, len(t)])
    for i, resmode in enumerate(response_qs_t):
        func_qs = InterpolatedUnivariateSpline(t_qs, resmode)
        resp_qs_intpl[i] = func_qs(t)

    # IRFFT of impulsive part
    p_sections = imp_discretize(impact_force_t)
    p_imp_f = pyfftw.interfaces.numpy_fft.rfft(p_sections, axis=1)
    # N/m over width (only works if sections are of constant size)

    # Compute response for modes
    response_imp = np.zeros([Gate.n_modes, len(Gate.f_trunc)], dtype=np.complex)
    np.einsum('jl,ijkl,k->kl', p_imp_f, frf_intpl, mode_shape,
              out = response_imp, dtype=np.complex)

    response_imp_t = pyfftw.interfaces.numpy_fft.irfft(response_imp, axis=1)
    response_imp_t = np.insert(response_imp_t, response_imp_t.shape[1], response_imp_t[:,-1], axis=1)

    # Combine responses and forces
    response_tot_t = resp_qs_intpl + response_imp_t
    return response_tot_t
