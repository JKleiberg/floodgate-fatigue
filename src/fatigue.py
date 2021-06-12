""" Functions to calculate and plot the fatigue damage for a stress time series.
"""
import rainflow
import numpy as np
import matplotlib.pyplot as plt
from src.utilities import PlotText
from src.pressure import pressure
from src.stress import stress_time
from src.configuration import gamma_Mf, gamma_Ff

# Characteristic values
m1 = 3
m2 = 5
Nc = 2e6
Nd = 5e6

def fatigue(response_t, cat):
    """Calculates the fatigue according to the Eurocode two-slope Miner method.

    Parameters:
    response_t: Time series of the gate stress at a certain point (Pa)
    cat: Detail category of material at coordinate (MPa)

    Returns:

    D: Fatigue damage factor (-)

    """
    amp, cycles = np.array(rainflow.count_cycles(response_t, nbins=None)).T

    # Palmgren-Miner
    loads = amp/10**6*gamma_Ff
    sigma_c = cat/gamma_Mf
    sigma_D = (2/5)**(1/3)*sigma_c
    sigma_L = (5/100)**(1/5)*sigma_D

    sig_i = loads[loads>sigma_D]
    n_i = cycles[loads>sigma_D]
    sig_j = loads[loads<=sigma_D]
    n_j = cycles[loads<=sigma_D]
    Q = sum(n_i*sig_i**3) + sum(n_j*sig_j**3*(sig_j/sigma_D)**2)
    D_eq = Q/(Nd*sigma_D**3)
    return D_eq

def sn_graph(response_t, cat):
    """Calculates the fatigue according to the Eurocode two-slope Miner method
       and plots the result in a graph

    Parameters:
    response_t: Time series of the gate stress at a certain point (Pa)
    cat: Detail category of material at coordinate (MPa)

    Returns:

    D: Fatigue damage factor (-)
    fig: Figure of fatigue distribution, S/n-line, and equivalent stress cycle.

    """
    amp, cycles = np.array(rainflow.count_cycles(response_t,nbins=500)).T
    loads = amp/10**6*gamma_Ff

    sigma_c = cat
    sigma_D = (2/5)**(1/3)*sigma_c
    sigma_L = (5/100)**(1/5)*sigma_D
    sig_i = loads[loads>sigma_D]
    n_i = cycles[loads>sigma_D]
    sig_j = loads[loads<=sigma_D]
    n_j = cycles[loads<=sigma_D]
    Q = (sum(n_i*sig_i**3) + sum(n_j*sig_j**3*(sig_j/sigma_D/gamma_Mf)**2))/(sigma_D/gamma_Mf)**3
    D_eq = Q/Nd
    
    print('Damage = '+str(round(100*D_eq,5))+'%')
    fig, ax = plt.subplots(figsize=(10,5))
    plt.vlines(Nc, min(loads), cat, ls='dotted', color="#c3312f")
    plt.barh(loads, cycles, height=np.diff(loads, append=loads[-1]),
             label='Load cycles', color="#00A6D6", alpha=0.25)
#     sn, = ax.loglog(Nr, loads, c="#c3312f", label='S-N line')
    categories = [36,40,45,50,56,63,71,80,90,100,112,125,140,160]
    N_cats = [10,2e6,5e6,1e8,10e10]
    for c in categories:
        if c == cat:
            alpha = 1
            label = 'S-N line'
        else:
            alpha = 0.1
            label= None
        sigD = (2/5)**(1/3)*c
        sigL = (5/100)**(1/5)*sigD
        S_cats = [(2e6*c**3/10)**(1/3), c, sigD, sigL, sigL]
        ax.loglog(N_cats, S_cats, c="#c3312f", alpha=alpha, label=label)
    plt.barh(cat, Q/sigma_c**3, height=5, label='Equivalent Stress Range', color="#00A6D6")
    plt.annotate('Detail category', xy=(Nc,cat), xycoords='data',
             xytext=(Nc/30,cat/2), arrowprops=dict(arrowstyle="->"))
    plt.annotate('Constant amplitude fatigue limit', xy=(Nd,sigma_D), xycoords='data',
                 xytext=(Nd/2,sigma_D*2), arrowprops=dict(arrowstyle="->"))
    plt.annotate('Cut-off limit', xy=(1e8,sigma_L), xycoords='data',
                 xytext=(1e8,sigma_L*2), arrowprops=dict(arrowstyle="->"))
    plt.ylim(10, max(max(loads), 300))
    plt.xlim(10, 10**10)
    plt.xticks(ticks=10**np.arange(10))
    plt.grid(lw=0.25)
    plt.ylabel('Stress range [MPa]')
    plt.xlabel('Number of cycles [-]')
    plt.legend(loc='upper left')
    sn_line = plt.gca().lines[categories.index(cat)]
    PlotText.line_annotate('m='+str(m1), sn_line, Nc/2)
    PlotText.line_annotate('m='+str(m2), sn_line, Nd*5)
    plt.close(fig)
    return D_eq, fig

def fatiguerepeater(cases):
    """Worker function for calculating the fatigue."""
    D_list = []
    for case in cases:
        f, pqs_f, p_imp_t, _,_ = pressure(U=case[0], hsea=case[1])
        _, response_t = stress_time(f, pqs_f, p_imp_t,case[3],case[4])
        D = fatigue(response_t)
        D_list.append({'U':case[0],'h':case[1],'D':D})
    return D_list

# def fatigueworker(args):
#     D_list = []
#     for case in args:
#         f, pqs_f, p_imp_t,_,_ = pressure(case[0][1], case[0][2])
#         _, response_tot_t = stress_time(f, pqs_f, p_imp_t, case[1])
#         D = fatigue(response_tot_t, 100)
#         D_list.append({'p':case[0][0], 'mean_u':case[0][1], 'mean_h':case[0][2],'D':D})
#     return D_list
