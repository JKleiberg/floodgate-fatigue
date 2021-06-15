""" Functions to evaluate the fatigue over the lifetime of the gate at a specified coordinate,
    and plot the results.
"""
import os
import numpy as np
import pandas as pd
import dill
import multiprocess
from src.pressure import pressure
from src.stress  import stress_time
from src.fatigue import fatigue
from src.configuration import hlife
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from mpl_toolkits.axes_grid1 import make_axes_locatable
TUred = "#c3312f"
TUblue = "#00A6D6"
TUgreen = "#00a390"
# Load system properties
with open('../data/06_transferfunctions/current_case.pkl', 'rb') as file:
    GATE = dill.load(file)

def fatigueworker(args):
    """Worker function for generate_fatigue.

    Parameters:
    args: List of case properties and coordinates for fatigue calculation

    Returns:

    D_list: List of fatigue damage factors for given load cases (-)

    """
    D_list = []
    for case in args:
        f, pqs_f, p_imp_t,_,_ = pressure(case[0][1], case[0][2])
        _, response_tot_t = stress_time(f, pqs_f, p_imp_t, case[1])
        D = fatigue(response_tot_t, 100)
        D_list.append({'p':case[0][0], 'mean_u':case[0][1], 'mean_h':case[0][2],'D':D})
    return D_list

def generate_fatigue(coords, version, N=1e6):
    """Generates a list of fatigue damage factors for a given gate coordinate and set of load cases.

    Parameters:
    coords: Coordinates at which fatigue is to be calculated (m)
    version: Version name under which results are stored
    N: Maximum number of load cases to evaluate (only change for tests) (-)

    Returns:

    all_damage: List of load cases with their probability of occurence,
                and associated fatigue damage factor (-)

    """
    print("Analyzing coordinate "+str(coords))
    # Load case list
    with open('../data/03_loadevents/filtered_cases.cp.pkl', 'rb') as f:
        N_cases = dill.load(f).head(int(N))
    # Fetch already calculated cases (if applicable)
    path = '../data/07_fatigue/%s/%s.pkl'%(GATE.case, version)
    if os.path.exists(path):
        print ("Using existing runs from %s" % path)
        with open(path, 'rb') as old_cases:
            oldcases = dill.load(old_cases)
    else:
        oldcases = pd.DataFrame(columns = ['p','mean_u','mean_h'])
        print ("Created new file at %s" % path)

    newcases = pd.concat([oldcases[['p','mean_u','mean_h']], N_cases]).drop_duplicates(
        keep=False, subset=['mean_u','mean_h'])[['p','mean_u','mean_h']]
    if len(newcases) == 0:
        return print('N load cases already exist.')
    # Choose number of cores and split input array. Shuffle load cases so each thread has equal work
    cores = multiprocess.cpu_count()
    print('Using '+str(cores)+' cores')
    args = [[case,coords] for case in newcases.sample(frac=1).to_numpy()]

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
    splitcases = chunks(args, cores)
    pool = multiprocess.Pool(cores)
    D_list = pool.map(fatigueworker, splitcases)
    pool.close()
    pool.join()

    new_damage = pd.DataFrame.from_records(np.concatenate(D_list))
    all_damage = pd.concat([oldcases,new_damage])
    with open(path, 'wb') as file:
        dill.dump(all_damage, file)
    print('Successfully performed %s fatigue analyses.'%len(new_damage))
    return all_damage

def simulations(runs, version, average_only=False, std=0.1):
    """Runs probabilistic simulations on previously computed list of fatigue load cases.

    Parameters:
    runs: Amount of simulations to run (-)
    version: Version name under which results are stored
    average_only: If True, the function only calculates the expected value,
                  rather than performing a Monte Carlo simulation.

    Returns:

    fig = Figure of cumulative fatigue development and relative importance of load cases.
    D_expected = Expected value of lifetime fatigue damage.
    totals = List of final lifetime fatigue values for all simulations.
    [maxes,means,mins]: List of maximum, mean, and minimum values
                        across all simulation at every point in time.

    """
    rng = np.random.default_rng()
    case_directory = '../data/07_fatigue/%s'%GATE.case
    if not os.path.exists(case_directory):
        os.mkdir(case_directory)
    with open('%s/%s.pkl' %(case_directory, version), 'rb') as f:
        cases_calc = dill.load(f).sort_values(['D'])
    cases_calc['damage'] = 100*cases_calc['D']*cases_calc['p'].to_list() # in %
    # Add 'empty' cases that were filtered out earlier
    cases = cases_calc.append({'mean_u':0,'mean_h':6,'D':0,'p':1-sum(cases_calc['p']), 'damage':0},
                              ignore_index=True)
    T = int(hlife*24*365.25)
    t = np.linspace(0,hlife,T)
    # Expected lifetime fatigue
    D_expected = sum(cases['damage'])*T
    if average_only:
        return D_expected
    
    runs_path = '%s/%s_lifetimes_v%s.pkl'%(case_directory, runs, version)
    if os.path.exists(runs_path):
        print("Using old simulations from %s"%runs_path)
        with open(runs_path, 'rb') as file:
            fig, D_expected, totals, [maxes,means,mins] = dill.load(file)
    else:
        print('Running %s Monte Carlo simulations...'%runs)
        # Randomly generate 100 years of hourly simulations, 'n' at a time
        n = 100  #chunk row size
        totals = []
        maxes = np.zeros(T)
        mins = np.ones(T)*10**6
        means = np.zeros(T)
        for i in range(int(np.ceil(runs/n))):
            events = rng.choice(cases['D'], [n,T], p=cases['p'])
            res = np.random.normal(events, std*events)*100 #in %
            cumulative=np.cumsum(res,axis=1)
            totals += np.max(cumulative,axis=1).tolist()
            means += cumulative.sum(axis=0)
            maxes = np.max(np.vstack((maxes,cumulative)),axis=0)
            mins  = np.min(np.vstack((mins,cumulative)),axis=0)
        means /= runs
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=[12,5])
    ax1.grid(alpha=0.5)
    ax1.hlines(100,0,100,ls='--', alpha=0.5, lw=3, color='#00A6D6')
    ax1.set_title('Damage accumulation for '+str(runs)+' simulations')
    ax1.set_xlim(0,hlife)
    ax1.set_ylim(0,120)
    ax1.plot(t,maxes,label='Upper bound', color='#c3312f')
    ax1.plot(t,means,label='Average', color='gray')
    ax1.plot(t,mins,label='Lower bound', color='#00A6D6')
    ax1.fill_between(t, mins, maxes, alpha=0.1, color='gray')
    ax1.set_xlabel('Time [years]')
    ax1.set_ylabel('Fatigue damage [%]')
    ax1.legend(loc='lower right')

    with open('../data/03_loadevents/currentbins.cp.pkl', 'rb') as f:
        x_u, x_h, u_bins, h_bins = dill.load(f)
    hist = ax2.hist2d(cases_calc['mean_h'], cases_calc['mean_u'], bins=[h_bins,u_bins],
                      range=[(min(x_h),max(x_h)),(min(x_u),max(x_u))],
                      weights=cases_calc['damage']*100, cmap='Blues', cmin=1e-6)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb = fig.colorbar(hist[3], cax=cax, orientation='vertical')
    cb.ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax2.set_title('Damage contribution per load case')
    ax2.set_xlabel('d [m]')
    ax2.set_ylabel('$U_{10}$ [m/s]')
    ax2.set_xticks(np.linspace(min(x_h), max(x_h), h_bins+1), minor=True)
    ax2.set_yticks(np.linspace(min(x_u), max(x_u), u_bins+1), minor=True)
    ax2.set_xlim(5,9)
    ax2.set_ylim(min(cases['mean_u']),50)
    with open(runs_path, 'wb') as f:
        dill.dump([fig, D_expected, totals, [maxes,means,mins]], f)
    plt.close(fig)
    return fig, D_expected, totals, [maxes,means,mins]

def compare_simulations(runs, cases, labels, colors=[TUblue, TUred, TUgreen, '#f1be3e', '#eb7245']):
    """Runs probabilistic simulations on previously computed list of fatigue load cases.

    Parameters:
    runs: Amount of simulations to run (-)
    cases: Version names under which results to be compared are stored
    labels: Labels for each the results to be compared.
    colors: Colors for each of the results to be compared.

    Returns:

    fig = Figure of comparison.

    """
    fig, ax1 = plt.subplots(figsize=[12,5])
    T = int(hlife*24*365.25)
    t = np.linspace(0, hlife, T)
    ax1.grid(alpha=0.5)
    ax1.hlines(100 , 0, 100,ls='--', alpha=0.5, lw=3, color='black', label='Fatigue capacity')
    ax1.set_xlim(0,hlife)
    ax1.set_ylim(0,120)
    total_list = []
    for case, color, label in zip(cases, colors, labels):
        _, _, totals, lines = simulations(runs, version=case, average_only=False)
        maxes, means, mins = lines
        ax1.plot(t, maxes, color=color, ls=':',alpha=0.7)
        ax1.plot(t, means, label=label, color=color)
        ax1.plot(t, mins, color=color, ls=':', alpha=0.7)
        ax1.fill_between(t, mins, maxes,alpha=0.1, color=color)
        total_list.append(totals)
    with open('../data/07_fatigue/%s/comparison_%s_lifetimes_%s.pkl'%(GATE.case, runs, labels), 'wb') as f:
        dill.dump(total_list, f)
    ax1.set_xlabel('Time [years]')
    ax1.set_ylabel('Fatigue damage [%]')
    ax1.legend(loc='lower right')
    with open('../data/00_figs/comparison_'+str(cases)+'.png', 'wb') as f:
        plt.savefig(f)
    plt.close(fig)
    return fig
