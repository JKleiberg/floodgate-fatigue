import numpy as np
import matplotlib.pyplot as plt
import pyfftw
from scipy.interpolate import interp1d
from src.configuration import N_HOURS, WIDTH, dz, H_GATE, Gate, z_coords, t
from src.woodandperegrine import woodandperegrine
pyfftw.config.NUM_THREADS = 4
TUred = "#c3312f"
TUblue = "#00A6D6"
TUgreen = "#00a390"

# Load FRF corresponding to loaded system properties
directory = '../data/06_transferfunctions/'+str(Gate.case)+'/FRF_'+str(Gate.n_modes)+'modes'+str(Gate.case)+'.npy'
try:
    frf_intpl = np.load(directory, mmap_mode='r')
    # Map mode only loads parts of 3Gb matrix when needed instead of keeping it all in RAM.
except OSError:
    print("An exception occurred: no FRF found at "+directory)

# Stress functions
def qs_disp_response(f, pqs_f, coords):
    # Interpolate to correct resolution
    FRF_qs = np.zeros((Gate.n_x,Gate.n_z,Gate.n_modes,len(f)),dtype='complex')
    for i in range(Gate.n_x):
        for j in range(Gate.n_z):
            for k in range(Gate.n_modes):
                func = interp1d(Gate.FRF_f,Gate.FRF[i][j][k],
                                bounds_error=False,fill_value=np.nan)
                FRF_qs[i,j,k,:] = func(f)

    # Integrate pressures over sections
    # does not integrate over x!
    pqs_sections = np.array([np.sum(np.split(pqs_f,Gate.ii_z[1:-1], axis=0)[i],axis=0)*dz\
                             /(z_coords[Gate.ii_z[i+1]]-z_coords[Gate.ii_z[i]])\
                             for i in range(Gate.n_z)])
    q_tot = pqs_sections.sum(axis=(0))*(H_GATE/Gate.n_z)
    # N/m (only works if sections are of constant size)

    # Compute response spectrum
    response_gate_qs = np.zeros((Gate.n_modes,len(f)), dtype=np.complex64)
    mode_shape = [[i] for i in Gate.disp3D.loc[coords].to_list()]
    np.multiply(np.multiply(pqs_sections[np.newaxis,:,np.newaxis,:],FRF_qs)\
                .sum(axis=(0,1)), mode_shape, out = response_gate_qs)
    return q_tot, response_gate_qs

def impulsive_disp_response(p_imp_t, coords):
    # Prepare Wood & Peregrine dimensionless impact shape
    Pz_impact_U1 = woodandperegrine()
    # Construct force-time matrix for all z-coordinates
    p_matrix = np.zeros((len(p_imp_t), len(Pz_impact_U1)))
    np.multiply.outer(p_imp_t, Pz_impact_U1, out = p_matrix)

    # does not integrate over x!
    p_sections = np.array([np.sum(np.split(p_matrix,Gate.ii_z[1:-1], axis=1)[i],axis=1)*dz\
                           /(z_coords[Gate.ii_z[i+1]]-z_coords[Gate.ii_z[i]]) for i in range(Gate.n_z)])

    # Find force spectrum for sections
    p_imp_f = pyfftw.interfaces.numpy_fft.rfft(p_sections,axis=1)

    # Compute response for modes
    response_gate_imp = np.zeros((Gate.n_modes,len(Gate.f_trunc)), dtype=np.complex64)
    mode_shape = [[i] for i in Gate.disp3D.loc[coords].to_list()]

    np.multiply(np.multiply(p_imp_f[np.newaxis,:,np.newaxis,:],frf_intpl)\
                .sum(axis=(0,1)), mode_shape, out = response_gate_imp)

    # Find total force
    F_gate = p_sections.sum(axis=(0))*(H_GATE/Gate.n_z)
    # N/m over width (only works if sections are of constant size)
    return F_gate, response_gate_imp

def disp_time(f, pqs_f, p_imp_t, coords,plot=False, plotrange=[50,80]):
    # IFFT of quasi-static part
    N = len(pqs_f[0])  # IRFFT scale factor
    q_tot, response_qs = qs_disp_response(f, pqs_f, coords)
    response_qs_t = N*pyfftw.interfaces.numpy_fft.irfft(np.array([mode[~np.isnan(mode)]\
                                                        for mode in response_qs]))
    t_qs = np.linspace(0,3600*N_HOURS,len(response_qs_t.sum(axis=0)))
    # Interpolate qs
    func_qs = interp1d(t_qs,response_qs_t.sum(axis=0),
                       bounds_error=False,fill_value=np.nan)
    resp_qs_intpl = func_qs(t)

    F_qs = pyfftw.interfaces.numpy_fft.irfft(q_tot)*N
    func_F_qs = interp1d(t_qs,F_qs, bounds_error=False,fill_value=np.nan)
    F_qs_intpl = func_F_qs(t)

    # IFFT of impulsive part
    F_gate, response_imp = impulsive_disp_response(p_imp_t, coords)
    response_imp_t = pyfftw.interfaces.numpy_fft.irfft(np.array([mode[~np.isnan(mode)]\
                     for mode in response_imp]),n=len(t)).sum(axis=0)

    # Combine responses and forces
    response_tot_t = resp_qs_intpl + response_imp_t
    F_tot = F_qs_intpl + F_gate

    if plot == True:
        fig, ax1 = plt.subplots(figsize=[15,5])
        ax2 = ax1.twinx()
        ax1.set_title('Load and displacement at X = '+str(coords[0])+'m,\
        Y = '+str(coords[1])+'m, Z = '+str(coords[2])+'m',fontsize = 14)
        ax1.set_xlim(plotrange[0],plotrange[1])
        ax1.set_xlabel('t [s]')
        section = np.where((t>plotrange[0]) & (t<plotrange[1]))[0]
        resp = ax1.plot(t[section],-1000*response_tot_t[section], label = 'Gate deflection',
                       color=TUblue) # minus sign so direction matches forces
        ax1.set_ylabel('Deflection [mm]', fontsize = 12)
        d_lim = 1.2*max(-response_tot_t[section])*1000
        d_mean = np.mean(-response_tot_t[section])*1000
        ax1.set_ylim(d_mean-d_lim, d_lim)
        ax1.legend()

        force = ax2.plot(t[section],F_tot[section]/1000, color=TUred, label = 'Wave force')
        ax2.set_ylabel('Integrated wave force [$kN/m$]', fontsize = 12)
        F_lim = 1.2*max(F_tot[section])/1000
        F_mean = np.mean(F_tot[section])/1000
        ax2.set_ylim(F_mean-F_lim, F_lim)
        lines = resp + force
        labs = [l.get_label() for l in lines]
        ax1.grid(lw=0.25)
        ax1.legend(lines,labs, fontsize = 12)
    return F_tot, response_tot_t

def disp_time_modes(f, pqs_f, p_imp_t, coords,plot=False, plotrange=[50,80]):
    # IFFT of quasi-static part
    N = len(pqs_f[0])  # IRFFT scale factor
    q_tot, response_qs = qs_disp_response(f, pqs_f, coords)
    response_qs_t = N*pyfftw.interfaces.numpy_fft.irfft(np.array([mode[~np.isnan(mode)]\
                                                        for mode in response_qs]))
    t_qs = np.linspace(0,3600*N_HOURS,N)
    # Interpolate qs
    func_qs = interp1d(t_qs,response_qs_t,
                       bounds_error=False,fill_value=np.nan)
    resp_qs_intpl = func_qs(t)

    F_qs = pyfftw.interfaces.numpy_fft.irfft(q_tot)*N
    func_F_qs = interp1d(t_qs,F_qs, bounds_error=False,fill_value=np.nan)
    F_qs_intpl = func_F_qs(t)

    # IFFT of impulsive part
    F_gate, response_imp = impulsive_disp_response(p_imp_t, coords)
    response_imp_t = pyfftw.interfaces.numpy_fft.irfft(np.array([mode[~np.isnan(mode)]\
                     for mode in response_imp]),n=len(t))

    # Combine responses and forces
    response_tot_t = resp_qs_intpl + response_imp_t
    F_tot = F_qs_intpl + F_gate

    if plot == True:
        fig, ax1 = plt.subplots(figsize=[15,5])
        ax2 = ax1.twinx()
        ax1.set_title('Load and displacement at X = '+str(coords[0])+'m,\
        Y = '+str(coords[1])+'m, Z = '+str(coords[2])+'m',fontsize = 14)
        ax1.set_xlim(plotrange[0],plotrange[1])
        ax1.set_xlabel('t [s]')
        resp = ax1.plot(t,-1000*response_tot_t, label = 'Gate deflection') # minus sign so direction matches forces
        ax1.set_ylabel('Deflection [mm]', fontsize = 12)
        plot_t = np.where((t>plotrange[0]) & (t<plotrange[1]))[0]
        d_lim = 1.2*max(response_tot_t[plot_t])*1000
        d_mean = np.mean(response_tot_t[plot_t])*1000
        ax1.set_ylim(d_mean-d_lim, d_lim)
        ax1.legend()

        force = ax2.plot(t,F_tot/1000, color='r', label = 'Wave force')
        ax2.set_ylabel('Integrated wave force [$kN/m$]', fontsize = 12)
        F_lim = 1.2*max(F_tot[plot_t])/1000
        F_mean = np.mean(F_tot[plot_t])/1000
        ax2.set_ylim(F_mean-F_lim, F_lim)

        lines = resp + force
        labs = [l.get_label() for l in lines]
        ax1.grid(lw=0.25)
        ax1.legend(lines,labs, fontsize = 12)
    return F_tot, response_tot_t
