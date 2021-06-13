"""Calculates the fatigue and contribution of each mode at every point in the gate"""
import os
import sys
import dill
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.ticker as mtick
import matplotlib.colors as colors
import pyfftw
from scipy.interpolate import interp1d
import multiprocess
pyfftw.config.NUM_THREADS = 4
root_dir = os.path.join(os.getcwd(), '..')
sys.path.append(root_dir)
from src.configuration import N_HOURS, t
from src.woodandperegrine import woodandperegrine
from src.stress import qs_discretize, imp_discretize, stress_per_mode
from src.utilities.nearest import nearest
from src.fatigue import fatigue
from src.pressure import pressure
# Load system properties
with open('../data/06_transferfunctions/current_case.pkl', 'rb') as file:
    GATE = dill.load(file)
# Load FRF corresponding to loaded system properties
directory = '../data/06_transferfunctions/'+str(GATE.case)+'/FRF_'+str(GATE.case)+'_'+str(GATE.n_modes)+'modes.npy'
try:
    frf_intpl = np.load(directory, mmap_mode='r')
    # Map mode only loads parts of 3Gb matrix when needed instead of keeping it all in RAM.
except OSError:
    print("An exception occurred: no FRF found at "+directory)

def fatigue_gate(u_wind, h_sea, cat, ID=''):
    """Calculates the fatigue and contribution of each mode at every point in the gate

    Parameters:
    u_wind: Average wind velocity (m/s)
    h_sea: Average water depth (m)
    cat: Fatigue detail category (MPa)
    ID: ID of load case for comparisons

    Returns:
    damage: List of fatigue damage at every point (-)
    modeshare: List of how much each mode contributed
               to the total response at a coordinate (%)

    """

    # Prepare Wood & Peregrine dimensionless impact shape
    Pz_impact_U1 = woodandperegrine(GATE)

    freqs, pqs_f, impact_force_t, Hs, Tp = pressure(u_wind, h_sea)
    length_scale = len(pqs_f[0])
    pqs_sections = qs_discretize(pqs_f)
    func = interp1d(GATE.f_tf, GATE.FRF, axis=3)
    FRF_qs = func(freqs)
    p_imp_f = pyfftw.interfaces.numpy_fft.rfft(imp_discretize(impact_force_t),axis=1)

    def divide_chunks(items, chunksize):
        for i in range(0, len(items), chunksize):
            yield items[i:i + chunksize]
    #Tradeoff between memory and processing time
    chunksize = 15
    cores = 4
    col_chunks = list(divide_chunks(GATE.stress_neg, chunksize))

    def gatefatigue_worker(chunk):
        response_qs = np.zeros([len(chunk), GATE.n_modes, len(freqs)], dtype=np.complex)
        np.einsum('jl,ijkl,pk->pkl', pqs_sections, FRF_qs, chunk,
                  out = response_qs, dtype=np.complex)
        int_qs = [np.trapz(abs(point), x=freqs, axis=1)*length_scale for point in response_qs]
        response_qs_t = length_scale*pyfftw.interfaces.numpy_fft.irfft(\
                        response_qs.sum(axis=1), axis=1)
        t_qs = np.linspace(0, 3600*N_HOURS, response_qs_t.shape[1])
        # Interpolate qs
        func_qs = interp1d(t_qs, response_qs_t, axis=1)
        resp_qs_intpl = func_qs(t)

        # Compute response for modes
        response_imp = np.zeros([len(chunk), GATE.n_modes, len(GATE.f_intpl)], dtype=np.complex)
        np.einsum('jl,ijkl,pk->pkl', p_imp_f, frf_intpl, chunk,
                  out=response_imp, dtype=np.complex)
        int_imp = [np.trapz(abs(point), x=GATE.f_intpl, axis=1) for point in response_imp]
        response_imp_t = pyfftw.interfaces.numpy_fft.irfft(response_imp.sum(axis=1), axis=1)
        response_imp_t = np.insert(response_imp_t,response_imp_t.shape[1],
                                   response_imp_t[:,-1], axis=1)
        response_tot_t = resp_qs_intpl + response_imp_t
        int_tot = np.array([(x + y) for x, y in zip(int_qs, int_imp)])
        modeshare = [i/sum(i) for i in int_tot]
        damage = [fatigue(res, cat) for res in response_tot_t]
        return damage, modeshare

    pool = multiprocess.Pool(cores)
    damage, modeshare = zip(*pool.map(gatefatigue_worker, col_chunks))
    pool.close()
    with open('../data/08_analysis/full_gate_fatigue/gatefatigue_('\
              +str(u_wind)+','+str(h_sea)+'_'+str(ID)+').cp.pkl', 'wb') as file:
        dill.dump([np.concatenate(damage),np.concatenate(modeshare)],file)
    return np.concatenate(damage), np.concatenate(modeshare)

def plot_fatigue_gate(cases, titles, u_wind, h_sea):
    """Plots the fatigue experienced by every point on the gate. Allows for multiple inputs for comparison.

    Parameters:
    cases: List of previously calculated cases to plot
    titles: Titles to apply to plots
    u_wind: Wind velocity for which cases were evaluated
    h_sea: Average water level for which cases were evaluated

    """
    fig = plt.figure(figsize=[15,12])
    plt.tight_layout()
    for i, ID in enumerate(cases):
        with open('../data/08_analysis/full_gate_fatigue/gatefatigue_('+str(u_wind)+','+str(h_sea)+'_'+str(ID)+').cp.pkl', 'rb') as f:
            damage_gate, modeshare = dill.load(f)
        ax = fig.add_subplot(1, len(cases), i+1, projection='3d')
        Zmin = min(damage_gate)
        Zmax = max(damage_gate)
        cmap = plt.cm.Reds
        norm = colors.PowerNorm(gamma=0.3)

        coords = []
        response = []
        for face in GATE.faces:
            coords.append(GATE.coords[face-1])
            response.append(damage_gate[face-1].mean())

        facets = Poly3DCollection(coords)
        facets.set_facecolor(cmap(norm(response)))
        ax.add_collection3d(facets)

#         cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm),fraction=0.02, pad=0)

        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.set_xlim3d(0, GATE.WIDTH)
        ax.set_ylim3d(-5,5)
        ax.set_zlim3d(0,7.5)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        ax.view_init(30, 40)
        ax.set_title(titles[i], fontsize=15)
        print('Maximum fatigue in %s: %s'%(ID, round(np.max(damage_gate),5)))
#         fig.subplots_adjust(wspace=0.1)
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, fraction=0.03,
                     ticks=[10**-x for x in range(10)], format='%.0e')
        cbar.ax.set_title("D [-]")
    plt.close(fig)
    return fig

def plot_modeshare_gate(modes, modeshare):
    """Plots the contribution of mode(s) to the total fatigue at every gate coordinate.

    Parameters:
    modes: The mode(s) whose contribution to fatigue should be plotted (-)
    modeshare: List of mode contributions for every gate coordinate (%)

    """

    # Determine % of stress caused by specified modes
    share = np.array([sum(point[mode-1]/sum(point)*100 for mode in modes)\
                      for point in modeshare])

    fig = plt.figure(figsize=[8,8])
    ax = Axes3D(fig)
    Zmin = 0
    Zmax = 100
    cmap = plt.cm.Reds
    norm = colors.Normalize(Zmin,Zmax)

    coords = []
    response = []
    for face in GATE.faces:
        coords.append(GATE.coords[face-1])
        response.append(share[face-1].mean())

    facets = Poly3DCollection(coords)
    facets.set_facecolor(cmap(norm(response)))
    ax.add_collection3d(facets)

    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm),fraction=0.03, pad=.1)
    cbar.set_label("Share of response [%]", rotation=270, labelpad=20)

    plotmodes = ', '.join(map(str, modes))
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_xlim3d(0, GATE.WIDTH)
    ax.set_ylim3d(-4,4)
    ax.set_zlim3d(0,8)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.set_box_aspect((GATE.WIDTH, 8, GATE.HEIGHT))
    ax.view_init(30, 50)
    plt.close(fig)
    return fig

def modalstresscontribution(modeshare, coords):
    """Histogram with ranges of mode contributions to all coordinates.

    Parameters:
    modeshare: List of mode contributions for every gate coordinate (%)

    """

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(15,5))
    average = np.sum(modeshare,axis=0)/np.sum(modeshare)*100
    cumulative = np.pad(np.cumsum(average),(1,0))
    d_lower = average-np.min([m/np.sum(m)*100 for m in modeshare],axis=0)
    d_upper = np.max([m/np.sum(m)*100 for m in modeshare],axis=0)-average
    ax1.bar(np.arange(GATE.n_modes)+1,average, yerr=[d_lower,d_upper],
           align='center', facecolor='white', edgecolor='black', capsize=3, label='Mode average')
    ax1.plot(np.arange(GATE.n_modes+1),cumulative, marker='D',
             color='black', alpha=0.6, label='Cumulative')
    ax1.set_xticks(np.arange(GATE.n_modes)+1)
    ax1.set_xlim(0, GATE.n_modes+1)

    ax1.set_xlabel('Mode')
    ax1.set_ylabel('Mode contribution')
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax1.grid(axis='y', alpha=0.25)
    ax1.legend(loc='right')

    index = GATE.coords.tolist().index(list(coords))
    cumulative = np.pad(np.cumsum(modeshare[index]*100),(1,0))
    ax2.plot(np.arange(GATE.n_modes+1),cumulative, marker='D',
             color='black', alpha=0.6, label='Cumulative')
    ax2.bar(np.arange(GATE.n_modes)+1, modeshare[index]*100,
            align='center', label='Mode', facecolor='white', edgecolor='black')
    ax2.set_xticks(np.arange(GATE.n_modes)+1)
    ax2.set_xlim(0, GATE.n_modes+1)
    ax2.set_xlabel('Mode')
    ax2.set_ylabel('Mode contribution')
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax2.grid(axis='y', alpha=0.25)
    ax2.legend(loc='right')
    plt.close(fig)
    return fig

def modalfatiguecontribution(wind, hsea, coords, cat=100, save=False):
    coords = nearest(*coords)
    f, pqs_f, p_imp_t, Hs, Tp = pressure(wind, hsea)
    res = stress_per_mode(f, pqs_f, p_imp_t, coords=coords)
    D_list = [fatigue(np.sum(res, axis=0)-mode, 100) for mode in res]
    total = fatigue(np.sum(res, axis=0), 100)
    shares = 1-D_list/total
    if save:
        with open('../data/07_fatigue/shares_'+str(GATE.case)+'_%smps.cp.pkl'%wind, 'wb') as f:
            dill.dump(shares, f)
    fig = plt.figure()
    modes = np.arange(16)+1
    plt.bar(modes, 100*shares, width=0.5, facecolor='white', edgecolor='black', label='Amount of fatigue damage lost by removing this mode')
    plt.ylim(0,100)
    plt.xlabel('Mode')
    plt.ylabel('%')
    plt.grid(lw=0.25, axis='y')
    plt.xticks(modes)
    plt.legend()
    plt.close()
    return fig, shares
