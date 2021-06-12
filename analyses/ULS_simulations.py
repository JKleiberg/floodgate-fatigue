"""This module runs simulations for the gate stress at a certain coordinate
to check the ULS requirement"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cloudpickle
import multiprocess
from src.pressure import pressure
from src.stress import stress_time
import scipy.stats as ss
root_dir = os.path.join(os.getcwd(), '..')
sys.path.append(root_dir)
TUred = "#c3312f"
TUblue = "#00A6D6"
TUgreen = "#00a390"

def uls_worker(args):
    """Helper function for stress_simulations"""
    h_sea, u_wind, coords, chunk = args
    max_s_list = []
    for _ in range(chunk):
        freqs, pqs_f, impact_force_t, _,_ = pressure(u_wind, h_sea)
        _, response_t = stress_time(freqs, pqs_f, impact_force_t, coords)
        max_s_list.append(np.max(response_t))
    return max_s_list

def ULS_simulations(h_sea, u_wind, coords, runs, version=''):
    """Runs a lot of stress simulations for the same input at a given coordinate,
    and stores the peak stress that occurs during each simulation.

    Parameters:
    h_sea: Average water depth (m)
    u_wind: Average wind velocity (m/s)
    coords: xyz-coordinates of point where stresses should be evaluated (m)
    runs: Amount of simulations to be run (-)
    version: version of simulation for storage and comparisons (-)

    Returns:
    max_s_list: List of maximum gate stresses for each simulation (Pa)

    """
    print('Running ULS analysis for case: '+str(version))
    cores = 4 #multiprocess.cpu_count()
    print('Using '+str(cores)+' cores')
    chunk = int(runs/cores)
    args = cores*[[h_sea, u_wind, coords, chunk]]
    pool = multiprocess.Pool(cores)
    max_s_list = pool.map(uls_worker, args)
    pool.close()
    pool.join()

    max_s_list = np.concatenate(max_s_list)
    path = '../data/08_analysis/ULS_stress'
    with open(path+'/maxstresses_'+str(runs)+'_v'+str(version)+'.cp.pkl', 'wb') as file:
        cloudpickle.dump(max_s_list, file)
    print('Successfully performed '+str(runs)+' simulations.')
    return max_s_list

def uls_plot(runs, f_yield, version, fit=False):
    """Plots the stress simulations and shows in how many cases the structure failed"""
    with open('../data/08_analysis/ULS_stress/maxstresses_'+str(runs)+'_v'\
              +str(version)+'.cp.pkl','rb') as file:
        max_stresses = cloudpickle.load(file)/10**6 #MPa
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=[15,5])
    fails = np.where(max_stresses>f_yield)
    successes = np.where(max_stresses<=f_yield)
    ax1.scatter(fails,max_stresses[fails], facecolors='none',
                edgecolors='#c3312f', label='Gate failures')
    ax1.scatter(successes,max_stresses[successes], facecolors='none',
                edgecolors='#00A6D6', label='Acceptable stresses')
    ax1.hlines(f_yield,0,runs,ls='--', color=TUred)
    ax1.legend()
    ax1.grid(alpha=0.25)
    ax1.set_xlim(0,runs)
    ax1.set_xlabel('Simulation [-]')
    ax1.set_ylabel('Maximum stress [MPa]')
    if len(max_stresses[successes])>0:
        rate = len(max_stresses[fails])/len(max_stresses[successes])
    else:
        rate = 1
    print('Gate failed in '+str(round(rate*100,2))+'% of cases. The requirement is 1%.')

    bins = np.arange(min(max_stresses)-(min(max_stresses)%5), np.ceil(max(max_stresses)/5)*5, 5)
    _, bins, patches = ax2.hist(max_stresses, bins, color=TUblue, density=True, label='Peak stresses')
    for i, bin_ in enumerate(bins):
        if (bin_ >= f_yield) & (i<len(patches)):
            patches[i].set_facecolor(TUred)
    ax2.axvline(f_yield, 0, 1, ls='--', color=TUred, label='Yield strength')
    ax2.set_xlabel('Maximum stress [MPa]')
    ax2.set_ylabel('Occurences [-]')

    dist = ss.gamma.fit(max_stresses)
    fitted = ss.gamma.pdf(bins, *dist)
    ax2.plot(bins, fitted, color=TUred, label='Lognormal fit', alpha=0.8)
    ax2.set_xlim(min(max_stresses), np.max([f_yield+10, *max_stresses]))
    print('Estimated failure rate {:.2E}'.format(1-ss.gamma.cdf(f_yield, *dist)))
    ax2.legend(loc='upper right')
    plt.close(fig)
    return fig
