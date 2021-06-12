"""Calculates the stress and the relative importance of each mode at every point in the gate"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.ticker as mtick
import matplotlib.colors as colors
import pyfftw
from scipy.interpolate import interp1d
import cloudpickle
import multiprocess
pyfftw.config.NUM_THREADS = 4
root_dir = os.path.join(os.getcwd(), '..')
sys.path.append(root_dir)
from src.configuration import N_HOURS, WIDTH, H_GATE, t, Gate
from src.woodandperegrine import woodandperegrine
from src.stress import qs_discretize, imp_discretize
from src.pressure import pressure

# Load FRF corresponding to loaded system properties
directory = '../data/06_transferfunctions/'+str(Gate.case)+'/FRF_'+str(Gate.n_modes)+'modes'+str(Gate.case)+'.npy'
try:
    frf_intpl = np.load(directory, mmap_mode='r')
    # Map mode only loads parts of 3Gb matrix when needed instead of keeping it all in RAM.
except OSError:
    print("An exception occurred: no FRF found at "+directory)

def stress_gate(u_wind, h_sea, ID):
    """Calculates the stress and relative importance of each mode at every point in the gate

    Parameters:
    u_wind: Average wind velocity (m/s)
    h_sea: Average water depth (m)
    ID: ID of load case for comparisons

    Returns:
    max_stress: List of maximum stresses at every point (-)
    modeshare: List of how much each mode contributed
               to the total response at a coordinate (%)

    """
    # Prepare Wood & Peregrine dimensionless impact shape
    Pz_impact_U1 = woodandperegrine()
    
    freqs, pqs_f, impact_force_t, Hs, Tp = pressure(u_wind, h_sea)
    length_scale = len(freqs)
    pqs_sections = qs_discretize(pqs_f)
    func = interp1d(Gate.FRF_f,Gate.FRF, axis=3)
    FRF_qs = func(freqs)
    p_imp_f = pyfftw.interfaces.numpy_fft.rfft(imp_discretize(impact_force_t),axis=1)

    def divide_chunks(items, chunksize):
        for i in range(0, len(items), chunksize):
            yield items[i:i + chunksize]
    #Tradeoff between memory and processing time
    chunksize = 15
    cores = 4
    col_chunks = list(divide_chunks(Gate.stressneglist.tolist(), chunksize))

    def gatestress_worker(chunk):
        response_qs = np.zeros([len(chunk),Gate.n_modes,len(freqs)], dtype=np.complex)
        np.einsum('jl,ijkl,pk->pkl', pqs_sections, FRF_qs, chunk,
                  out = response_qs, dtype=np.complex)
        int_qs = [np.trapz(abs(point),x=freqs, axis=1)*length_scale for point in response_qs]
        response_qs_t = length_scale*pyfftw.interfaces.numpy_fft.irfft(\
                        response_qs.sum(axis=1), axis=1)
        t_qs = np.linspace(0,3600*N_HOURS,response_qs_t.shape[1])
        # Interpolate qs
        func_qs = interp1d(t_qs, response_qs_t, axis=1)
        resp_qs_intpl = func_qs(t)

        # Compute response for modes
        response_imp = np.zeros([len(chunk),Gate.n_modes,len(Gate.f_trunc)], dtype=np.complex)
        np.einsum('jl,ijkl,pk->pkl',p_imp_f,frf_intpl,chunk,
                  out = response_imp, dtype=np.complex)
        int_imp = [np.trapz(abs(point),x=Gate.f_trunc, axis=1) for point in response_imp]
        response_imp_t = pyfftw.interfaces.numpy_fft.irfft(response_imp.sum(axis=1), axis=1)
        response_imp_t = np.insert(response_imp_t,response_imp_t.shape[1],
                                   response_imp_t[:,-1],axis=1)
        response_tot_t = resp_qs_intpl + response_imp_t
        int_tot = np.array([(x + y) for x, y in zip(int_qs, int_imp)])
        modeshare = [i/sum(i) for i in int_tot]
        max_stress = [max(res) for res in response_tot_t]
        return max_stress, modeshare

    pool = multiprocess.Pool(cores)
    max_stress, modeshare = zip(*pool.map(gatestress_worker, col_chunks))
    pool.close()
    with open('../data/08_analysis/full_gate_stress/fullgate_stress_('\
              +str(u_wind)+','+str(h_sea)+'_'+str(ID)+').cp.pkl', 'wb') as file:
        cloudpickle.dump([np.concatenate(max_stress),np.concatenate(modeshare)], file)
    return np.concatenate(max_stress), np.concatenate(modeshare)

def plot_stress_gate(max_stress):
    """Plots the maximum stress experienced by every point on the gate.

    Parameters:
    max_stress: List of maximum stress values for every gate coordinate (Pa)

    """
    max_stress /= 10**6
    fig = plt.figure(figsize=[12,8])
    ax = Axes3D(fig)

    Zmin = min(max_stress)
    Zmax = max(max_stress)
    cmap = plt.cm.Reds
    norm = colors.PowerNorm(gamma=1)

    coords = []
    response = []
    for face in Gate.faces:
        coords.append(Gate.verts[face-1])
        response.append(max_stress[face-1].mean())

    facets = Poly3DCollection(coords)
    facets.set_facecolor(cmap(norm(response)))
    ax.add_collection3d(facets)

    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm),fraction=0.02, pad=0)
    cbar.set_label("Stress [MPa]", rotation=270, labelpad=20)

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_xlim3d(0, WIDTH)
    ax.set_ylim3d(-4,4)
    ax.set_zlim3d(0,8)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    x_scale=WIDTH
    y_scale=8
    z_scale=8
    scale=np.diag([x_scale, y_scale, z_scale, 1.0])
    scale=scale*(1.0/scale.max())
    scale[3,3]=1.0

    def short_proj():
        return np.dot(Axes3D.get_proj(ax), scale)
    ax.get_proj=short_proj
    ax.view_init(30, 50)
    plt.close(fig)
    return fig
