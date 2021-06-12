""" Generates either a JONSWAP- or a TMA-spectrum,
based on the wind velocity and average water depth.

"""
import random
import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from src.configuration import g, N_HOURS,F, C_ts

def jonswap(u_wind, depth, timestep):
    """Generates a JONSWAP spectrum based on the wind velocity,
    the average water depth, and a timestep provided by the generator.

    Parameters:
    u_wind: Wind velocity of load event (m/s)
    depth:  Average water depth during load event (m)
    timestep: Desired timestep of final water level series (s)

    Returns:
    freqs: Frequencies of JONSWAP spectrum (Hz)
    amp: Amplitude spectrum values (m)
    k: Wave numbers (rad/m)

    """

    ## Define frequencies
    f0 = 1/(N_HOURS*3600)
    freqs = np.fft.rfftfreq(int(N_HOURS*3600/timestep), timestep)

    ## Determine k for each frequency. Made faster with Airy wave theory
    def wavenumber(x,f):
        return (2*np.pi*f)**2 - x*g*np.tanh(x*depth)
    shallow_limit = np.sqrt(2*np.pi*g/(20*depth)*np.tanh(2*np.pi/20))/(2*np.pi)
    deep_limit = np.sqrt(np.pi*g/(depth)*np.tanh(np.pi))/(2*np.pi)

    k = np.zeros(len(freqs))
    for i,f in enumerate(freqs):
        if f < 0.5*shallow_limit:      # Factor 0.5 to make transition smoother
            k[i] = 2*np.pi*f/np.sqrt(g*depth)
        elif f > 1.5*deep_limit:       # Factor 1.5 to make transition smoother
            k[i] = (2*np.pi*f)**2/g
        else:
            k[i] = root(wavenumber, 0.01, args=(f)).x[0]

    if u_wind == 0:
        return freqs,np.zeros(len(freqs)),k

    F_dim = g*F/u_wind**2             # Dimensionless fetch
    f_p_dim = 2.18*F_dim**-0.27  # Dimensionless peak frequency (Kahma and Calkoen 1992)
    f_p = f_p_dim*g/u_wind            # peak frequency

    # Lewis and Allos, 1990
    alpha = 0.0317*f_p_dim**0.67
    gamma = 5.87*f_p_dim**0.86
    sigma_a = 0.0547*f_p_dim**0.32
    sigma_b = 0.0783*f_p_dim**0.16

    # Creating spectrum
    amp = np.zeros(len(freqs), dtype='complex')
    phase = [random.uniform(-np.pi,np.pi) for _ in range(len(freqs))] # Generate random phases
    for i,f in enumerate(freqs):
        if f == 0:
            continue
        if f > f_p:
            sigma = sigma_a
        else:
            sigma = sigma_b
        pm_shape = alpha*g**2*(2*np.pi)**-4*f**-5*np.exp(-5/4*(f/f_p)**-4)
        S = pm_shape*gamma**np.exp(-0.5*((f/f_p-1)/sigma)**2)
        spec =  np.sqrt(S*f0*2)                 # From power density to amplitude
        amp[i] = spec*np.exp(1j*phase[i])       # Add random phase
    return freqs, amp, k

def TMA(f, amp_jon, k):
    """Creates a TMA spectrum from a JONSWAP-spectrum with the depth correction factor phi.

    Parameters:
    f: Frequencies of JONSWAP spectrum (Hz)
    amp_jon: Amplitude spectrum values of JONSWAP spectrum (m)
    k: Wave numbers (rad/m)

    Returns:
    amp_tma: New amplitude spectrum values (m)

    """

    k_inf = (2*np.pi*f)**2/g             # k for infinite depth
    dk_inf = 2*(2*np.pi)**2*f/g          # df/dk for intinite depth
    dk_d = np.gradient(k, f)             # df/dk for inputs
    phi = np.true_divide(k_inf,k, out = np.zeros_like(k), where=k>0)**3\
    *np.true_divide(dk_d,dk_inf, out = np.zeros_like(k), where=dk_inf>0)
    amp_tma = phi*amp_jon
    return amp_tma

def spectrum_generator(u_wind, depth, spec='TMA'):
    """Generates a spectrum from the wind velocity and average water depth.
    Defaults to the TMA-spectrum, but an unmodified JONSWAP spectrum can be
    created if desired.

    Parameters:
    u_wind: Wind velocity of load event (m/s)
    depth:  Average water depth of load event (m)
    spec: Spectrum type ("TMA"/"JONSWAP")

    Returns:
    f: Spectrum frequencies (Hz)
    amp_tma: New amplitude spectrum values (m)
    k: Wave numbers (rad/m)
    Hm0: Significant wave height (m)
    Tp: Peak period (s)

    """

    F_dim = g*F/u_wind**2             # Dimensionless fetch
    f_p_dim = 2.18*F_dim**-0.27       # Dimensionless peak frequency (Kahma and Calkoen 1992)
    f_p = f_p_dim*g/u_wind            # peak frequency
    Tp = 1/f_p
    timestep = round(Tp/C_ts,3)
    f, amp, k = jonswap(u_wind, depth, timestep)
    if spec == 'TMA':
        amp = TMA(f,amp,k)
        Tp = 1/f[np.argmax(abs(amp))]
    elif spec != 'JONSWAP':
        print('error, wrong spectral type. Defaulting to JONSWAP')
    f0 = 1/(N_HOURS*3600)
    Hm0 = 4*np.sqrt(np.trapz(abs(amp)**2/(2*f0),f)) #4*sqrt(m0)
    return f, amp, k, Hm0, Tp

def plot_spectra(u_list, depth):
    """Generates and plots spectra from the wind velocity and average water depth.
    First plots a list of JONSWAP spectra, and then compares this to a TMA-spectrum with the same inputs.

    Parameters:
    u_list: List of wind velocities (m/s)
    depth:  Average water depth of load event (m)
    spec: Spectrum type ("TMA"/"JONSWAP")

    Returns:
    fig1, figs2: Plots of the wave spectra

    """
    f0 = 1/(N_HOURS*3600)
    fig1 = plt.figure()
    cm = plt.get_cmap('Blues') 
    scalarMap = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=-20, vmax=max(u_list)), cmap=cm)

    for U in u_list:
        f, amp, k,Hs,Tp = spectrum_generator(U, depth, spec = 'JONSWAP')
        spec = amp**2/(2*f0)
        plt.plot(f,abs(spec),label = str(U)+' m/s', c = scalarMap.to_rgba(U))
        plt.vlines(Tp**-1,0,max(abs(spec)), ls='--', color = plt.gca().lines[-1].get_color())

    plt.xlim(0,0.8)    
    plt.ylabel('$E_J$ [$m^2/Hz$]')
    plt.xlabel('f [Hz]')
    plt.legend()
    
    U = u_list[0]
    f, amp, k, Hs, Tp = spectrum_generator(U, depth, spec = 'JONSWAP')
    spec = amp**2/(2*f0)
    print("For an example load event with u_10="+str(U)+" m/s:")
    print("JONSWAP: Tp = "+str(round(Tp,2))+"s, Hs = "+str(round(Hs,2))+"m")
    ft, ampt, kt, Hs, Tp = spectrum_generator(U,depth,spec = 'TMA')
    spect = ampt**2/(2*f0)
    print("TMA: Tp = "+str(round(Tp,2))+"s, Hs = "+str(round(Hs,2))+"m")

    fig2 = plt.figure(figsize=(15,5))
    plt.plot(f, abs(spec)/max(abs(spec)), label='JONSWAP', color='#000000')
    plt.plot(ft, abs(spect)/max(abs(spec)), label='TMA', color='#00A6D6');
    ph = np.true_divide(abs(spect), abs(spec), out=np.zeros_like(k), where=abs(spec)>0)
    plt.plot(f, ph, label='$\phi(f,d)$', ls='--', color='#c3312f')
#     plt.title('Normalized wave spectra for d = '+str(depth)+'m and $U_{10}$ = '+str(U)+'m/s')
    plt.ylabel('$S/S_{max}$ [-]')
    plt.xlabel('f [Hz]')
    plt.xlim(0, 1)
    plt.legend()
    
    return fig1, fig2
