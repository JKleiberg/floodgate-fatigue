# import numpy as np
# import pandas as pd
# import cloudpickle
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable 
# import matplotlib.ticker as mtick
# from src.configuration import hlife, Gate
# TUred = "#c3312f"
# TUblue = "#00A6D6"
# TUgreen = "#00a390"

# def simulations(runs, version, average_only=False, std=0.1):
#     rng = np.random.default_rng()
#     with open('../data/07_fatigue/%s.cp.pkl' % version, 'rb') as f:
#         cases_calc = cloudpickle.load(f).sort_values(['D'])
#     cases_calc['damage'] = 100*cases_calc['D']*cases_calc['p'].to_list() # in %
#     # Add 'empty' cases that were filtered out earlier
#     cases = cases_calc.append({'mean_u':0, 'mean_h':6, 'D':0, 'p':1-sum(cases_calc['p']), 'damage':0}, ignore_index=True)
#     T = int(hlife*24*365.25)
#     t = np.linspace(0,hlife,T)
#     # Expected lifetime fatigue
#     D_expected = sum(cases['damage'])*T
#     if average_only:
#         return D_expected
    
#     # Randomly generate 100 years of hourly simulations, 'n' at a time
#     n = 100  #chunk row size
#     totals = []
#     maxes = np.zeros(T)
#     mins = np.ones(T)*10**6
#     means = np.zeros(T)
#     for i in range(int(np.ceil(runs/n))):
#         events = rng.choice(cases['D'], [n,T], p=cases['p'])
#         res = np.random.normal(events, std*events)*100 #in %
#         cumulative=np.cumsum(res,axis=1)
#         totals += np.max(cumulative,axis=1).tolist()
#         means += cumulative.sum(axis=0)
#         maxes = np.max(np.vstack((maxes,cumulative)),axis=0)
#         mins  = np.min(np.vstack((mins,cumulative)),axis=0)
#     means /= runs
#     fig, (ax1,ax2) = plt.subplots(1,2,figsize=[12,5])
#     ax1.grid(alpha=0.5)
#     ax1.hlines(100,0,100,ls='--', alpha=0.5, lw=3, color='#00A6D6');
#     ax1.set_title('Damage accumulation for '+str(runs)+' simulations')
#     ax1.set_xlim(0,hlife)
#     ax1.set_ylim(0,120)
#     ax1.plot(t,maxes,label='Upper bound', color='#c3312f')
#     ax1.plot(t,means,label='Average', color='gray')
#     ax1.plot(t,mins,label='Lower bound', color='#00A6D6')
#     ax1.fill_between(t, mins, maxes, alpha=0.1, color='gray')
#     ax1.set_xlabel('Time [years]')
#     ax1.set_ylabel('Fatigue damage [%]')
#     ax1.legend(loc='lower right')

#     with open('../data/03_loadevents/currentbins.cp.pkl', 'rb') as f:
#         x_u, x_h, u_bins, h_bins = cloudpickle.load(f)
#     hist = ax2.hist2d(cases_calc['mean_h'], cases_calc['mean_u'], bins=[h_bins,u_bins],
#                       range=[(min(x_h),max(x_h)),(min(x_u),max(x_u))],
#                       weights=cases_calc['damage']*100, cmap='Blues', cmin=1e-6) 
#     divider = make_axes_locatable(ax2)
#     cax = divider.append_axes('right', size='5%', pad=0.05)
#     cb = fig.colorbar(hist[3], cax=cax, orientation='vertical')
#     cb.ax.yaxis.set_major_formatter(mtick.PercentFormatter())
#     ax2.set_title('Damage contribution per load case')
#     ax2.set_xlabel('d [m]')
#     ax2.set_ylabel('$U_{10}$ [m/s]')
#     ax2.set_xticks(np.linspace(min(x_h), max(x_h), h_bins+1), minor=True)
#     ax2.set_yticks(np.linspace(min(x_u), max(x_u), u_bins+1), minor=True)
#     ax2.set_xlim(5,9)
#     ax2.set_ylim(min(cases['mean_u']),50)
#     with open('../data/07_fatigue/'+str(runs)+'_lifetimes_v'+str(version)+'.cp.pkl', 'wb') as f:
#         cloudpickle.dump([fig, D_expected, totals, [maxes,means,mins]], f)
#     plt.close(fig)
#     return fig, D_expected, totals, [maxes,means,mins]

# def compare_simulations(runs, cases, labels, color=[TUblue, TUred, TUgreen, '#f1be3e', '#eb7245']):
#     fig, ax1 = plt.subplots(figsize=[12,5])
#     T = int(hlife*24*365.25)
#     t = np.linspace(0, hlife, T)
#     ax1.grid(alpha=0.5)
#     ax1.hlines(100 , 0, 100,ls='--', alpha=0.5, lw=3, color='black', label='Fatigue capacity')
#     ax1.set_xlim(0,hlife)
#     ax1.set_ylim(0,120)
#     total_list = []
#     for i in range(len(cases)):
#         _, _, totals, lines = simulations(runs, version=cases[i], average_only=False)
#         maxes, means, mins = lines
#         upp, = ax1.plot(t, maxes, color=color[i], ls=':',alpha=0.7)
#         ax1.plot(t, means, label=labels[i], color=color[i])
#         ax1.plot(t, mins, color=color[i], ls=':', alpha=0.7)
#         ax1.fill_between(t, mins, maxes,alpha=0.1, color=color[i])
#         total_list.append(totals)
#     with open('../data/07_fatigue/comparison_'+str(runs)+"_lifetimes_"\
#               +str(labels)+'.cp.pkl', 'wb') as f:
#         cloudpickle.dump(total_list, f)
#     ax1.set_xlabel('Time [years]')
#     ax1.set_ylabel('Fatigue damage [%]')
#     ax1.legend(loc='lower right')
#     with open('../data/00_figs/comparison_'+str(cases)+'.png', 'wb') as f:
#         plt.savefig(f)
#     plt.close(fig)
#     return fig
