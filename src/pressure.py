"""This script computes the pressure on the gate surface due to a given wave spectrum"""
import dill
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from src.spec import spectrum_generator
from src.configuration import g, rho, cr, dt, t, H_LAKE, N_HOURS,\
                              tri_min, tri_mode, tri_max, beta_std, beta_mean
with open('../data/06_transferfunctions/current_case.pkl', 'rb') as file:
    GATE = dill.load(file)

def hydrostatic(hsea):
    """Calculates the net hydrostatic water pressure on the gate."""
    fhs_z     = []        # Hydrostatic pressure as function of z
    for i,z in enumerate(GATE.z_coords):
        if z < H_LAKE:
            fhs_lake = rho*g*(H_LAKE - z)
        else:
            fhs_lake = 0
        if z < hsea:
            fhs_sea = rho*g*(hsea - z)
        else:
            fhs_sea = 0
        fhs_z.append(fhs_sea-fhs_lake)
    return fhs_z

def linearwavetheory(amp_gate, k, h_sea):
    """Calculates the quasi-static wave pressure on the gate from a wave spectrum.

    Parameters:
    amp_gate: Reflected water level spectrum at the gate (m)
    k:  Wave numbers of spectrum (rad/m)
    h_sea: Average sea water level (m)

    Returns:
    fqs_fz: Quasi-static pressure spectrum (N/m2)

    """

    # Split spectrum based on whether it is deep, shallow, or intermediate water depth
    # Multiplied transition k's by factor to prevent discontinuities.
    shallow_k = 0.5*np.pi/(10*h_sea)
    deep_k = 2*np.pi/h_sea

    # Define precise function, and approximations for deep and shallow water (helps prevent overflow errors in cosh)
    def shallow(a):
        return a*rho*g
    def intermediate(a, k, z):
        return a*rho*g*np.cosh(k*z)/np.cosh(k*h_sea)
    def deep(a, k, z):
        return a*rho*g*np.exp(k*(z-h_sea))
    amp_shallow = amp_gate[k<shallow_k]
    amp_inter = amp_gate[(k>=shallow_k) & (k<=deep_k)]
    k_inter = k[(k>=shallow_k) & (k<=deep_k)]
    amp_deep = amp_gate[k>deep_k]
    k_deep = k[k>deep_k]

    ## Quasi-static pressure
    fqs_fz    = []
    for j, z in enumerate(GATE.z_coords):
        fqs_z =[]
        if z <= h_sea:
            p_shallow = shallow(amp_shallow)
            p_inter = intermediate(amp_inter, k_inter, z)
            p_deep = deep(amp_deep, k_deep, z)
            fqs_z = np.concatenate([p_shallow,p_inter,p_deep])
        else:
            fqs_z = np.zeros(len(k))
        fqs_fz.append(fqs_z)
    return np.array(fqs_fz)

def impactloads(wl_tot, tau_par):
    """Calculates the highly dynamic wave pressure on the gate due to
    impacts on the overhang.

    Parameters:
    wl_tot: Time series of water surface (m)

    Returns:
    impact_force_t: Time series of impact forces (N)

    """

    ## Interpolate and determine impact velocities
    t_eta = np.linspace(0,3600*N_HOURS,len(wl_tot))
    spl = InterpolatedUnivariateSpline(t_eta,wl_tot)
    wl_interpolated = spl(t)
    ## Find zero crossings and impact velocities
    pos = np.array(wl_interpolated < GATE.HEIGHT)
    crossings = (pos[:-1] & ~pos[1:]).nonzero()[0]
    impact_vel = np.diff(wl_interpolated)[crossings]/dt
    ## Generate probabilistic wave impact parameters and create wave forces
    tau = np.random.triangular(tau_par[0], tau_par[1], tau_par[2], size=len(impact_vel))
    beta = np.random.normal(beta_mean, beta_std, size=len(impact_vel))
    impact_vel = impact_vel/(tau*0.5)*beta
    impact_force_t = np.zeros(len(t))
    for i, crossing in enumerate(crossings):
        steps = np.int(np.ceil(tau[i]/dt/2)*2+1)
        slope_up = np.linspace(0,impact_vel[i],int((steps+1)/2))
        slope_down = np.linspace(impact_vel[i],0,int((steps+1)/2))
        pulse = np.concatenate([slope_up,slope_down[1:]])
        if (crossing+steps) < len(wl_interpolated)-1:
            impact_force_t[crossing:int(crossing+steps)] = pulse
    return impact_force_t

def pressure(u_wind, hsea, tau_par=[0.01,0.105,0.2], spec='TMA'):
    """Calculates the static, quasi-static, and highly dynamic wave pressures
    on the gate due to water level fluctuations and impacts on the overhang.

    Parameters:
    u_wind: Average wind velocity during load event (m/s)
    hsea: Average water level during load event (m)

    Returns:
    freqs: Frequencies of JONSWAP spectrum (Hz)
    pqs_f: Quasi-static pressure spectrum (N/m2)
    impact_force_t: Time series of impact forces (N)
    Hs: Significant wave height (m)
    Tp: Peak period (s)

    """

    h_sea_r = round(hsea,1) # so it fits mesh
    ## Generate JONSWAP spectrum
    freqs, amp, k, Hs, Tp = spectrum_generator(u_wind, h_sea_r, spec)
    amp_gate = (1+cr)*amp
    eta = np.fft.irfft(amp_gate)*len(k)
    pqs_f = linearwavetheory(amp_gate, k, h_sea_r)
    fhs_z = hydrostatic(h_sea_r)
    pqs_f[:,0] = fhs_z              # Add hydrostatic pressure at zero frequency

    wl_tot = eta + h_sea_r  # Add depth to wave fluctuations
    if (wl_tot < GATE.HEIGHT).all():
        return freqs, pqs_f, np.zeros(int(3600*N_HOURS/dt+1)), Hs, Tp
    impact_force_t = impactloads(wl_tot, tau_par)
    return freqs, pqs_f, impact_force_t, Hs, Tp
