"""Functions to import and pre-process the raw data to obtain a set of load case
that characterize the environmental conditions during the lifetime of the structure."""
import numpy as np
import pandas as pd
import cloudpickle
import scipy, scipy.signal
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from src.configuration import h0, cr
from src.spec import spectrum_generator

def import_raw_data(windfile, waterfile, bins=30):
    """Imports the raw historical data, formats it into a DataFrame,
       and plots the distributions.

    Parameters:
    windfile: Name of file containing wind data
    waterfile: Name of file containing water level data

    Returns:

    raw_data: DataFrame containing all the raw data
    fig: Figure of distributions of raw data

    """
    wind = pd.read_csv('../data/01_raw/%s'%windfile,skiprows =13, header = None, low_memory=False)
    wind["hour"] = wind[1].astype(str) + (wind[2]-1).astype(str)
    dateformat = "%Y%m%d%H"
    times = pd.to_datetime(wind['hour'], format=dateformat)
    wind.set_index(times, inplace=True)
    wind = wind.rename(columns={3: "wind"}).drop([0,1,2,'hour'], axis=1).apply(
           pd.to_numeric, errors='coerce').dropna()/10

    with open('../data/01_raw/%s'%waterfile) as raw:
        h = pd.read_csv(raw, usecols = ['NUMERIEKEWAARDE','WAARNEMINGDATUM','WAARNEMINGTIJD'],
                        delimiter=';').rename(columns={'NUMERIEKEWAARDE':"h",
                                                       'WAARNEMINGDATUM':"date",
                                                       'WAARNEMINGTIJD': "hour"
                                                      })
    times = pd.to_datetime(h['date'].astype(str) +'-'+ h['hour'].astype(str),
                           format="%d-%m-%Y-%H:%M:%S")
    h.set_index(times, inplace=True)
    h = h.drop(h[h.h > 10**5].index).drop_duplicates() #Filter bad data
    wind['h'] = h['h']/100 + h0
    wind = wind.dropna()
    raw_data = wind[(wind.wind != 0)]
    with open('../data/02_preprocessing/data_1971.cp.pkl', 'wb') as prepped:
        cloudpickle.dump(raw_data, prepped)
        print('Data stored in ../data/02_preprocessing/data_1971.cp.pkl')

    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.hist(raw_data['wind'], bins=bins, density='normed', edgecolor='black', facecolor='white')
    ax1.grid(lw=0.25)
    ax1.set_xlim(left=0)
    ax1.set_xlabel('$U_{10}$ [m/s]')
    ax1.set_ylabel('Normalized density [-]')

    ax2.hist(raw_data['h'], bins=bins, density='normed', edgecolor='black', facecolor='white')
    ax2.set_xlabel('$h_S$ [m]')
    ax2.set_ylabel('Normalized density [-]')
    ax2.grid(lw=0.25)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    fig.tight_layout()
    plt.close()
    return fig, raw_data

def create_pdf(data, rise=1, n_h=1000, n_u=1e4, h_range=[0,10]):
    """Creates climate change-adjusted probability distributions from raw data.

    Parameters:
    data: Formatted raw data
    rise: Amount of sea level rise at end of lifetime (m)
    n_h: Number of point at which water level pdf should be evaluated (-)
    n_u: Number of point at which wind velocity pdf should be evaluated (-)
    h_range: Range for water level distribution (m)

    Returns:

    fig: Figure showing the pre- and post-adjustment distributions
    x_h: X-axis of water level distribution
    hcc_pdf: Pdf values at points x_h
    x_u: X-axis of wind velocity distribution
    u_pdf: Pdf values at points x_u

    """
    ## Correlation
    corr, _ = scipy.stats.pearsonr(data['h'], data['wind'])
    print('Pearsons correlation: %.3f' % corr)
    ## Water level
    h_dist = scipy.stats.gaussian_kde(data['h']-h0)
    x_h_p = np.linspace(-h0, h0, int(n_h*(2*h0)/(h_range[1]-h_range[0])))
    delta = (x_h_p[1]-x_h_p[0])
    h_pdf = h_dist(x_h_p)
    cc_pdf = scipy.stats.uniform(0, rise)
    hcc_pdf = scipy.signal.fftconvolve(delta*h_pdf, delta*cc_pdf.pdf(x_h_p), 'same')/delta
    print("Integral of convoluted pdf: "+str(round(np.trapz(hcc_pdf, x_h_p), 3)))
    x_h = np.linspace(*h_range, n_h)
    hcc_pdf = abs(scipy.interpolate.interp1d(x_h_p+h0, hcc_pdf, kind='cubic')(x_h))
    h_pdf = scipy.interpolate.interp1d(x_h_p+h0, h_pdf)(x_h)
    fig = plt.figure()
    plt.plot(x_h-h0, cc_pdf.pdf(x_h-h0), label='Climate change pdf')
    plt.plot(x_h-h0, h_pdf, ls='--' , label='Raw data KDE')
    plt.plot(x_h-h0, hcc_pdf, color='red' , label='Convoluted pdf')
    plt.ylim(0, 1.005)
    plt.xlabel('Water level [m]')
    plt.ylabel('Probability density [-]')
    plt.grid(lw=0.25)
    plt.legend()
    plt.close()
    ## Wind
    u_dist = scipy.stats.lognorm.fit(data['wind'])
    prob = (10000*365.25*24)**-1 #Once in 10000 year event
    u10000 = scipy.stats.lognorm(*u_dist).ppf(1-prob)
    x_u = np.linspace(0, u10000, int(n_u))
    u_pdf = scipy.stats.lognorm.pdf(x_u, *u_dist)
    with open('../data/03_loadevents/pdfs.cp.pkl', 'wb') as file:
        cloudpickle.dump([x_h, hcc_pdf, x_u, u_pdf], file)
    return fig, [x_h, hcc_pdf, x_u, u_pdf]

def binning(dists, h_res=0.1, u_res=1, save=True):
    """Creates discrete load cases by integrating the probability distributions over small segments.

    Parameters:
    dists: Distributions derived from create_pdf
    h_res: Segment size for water level distribution (m)
    u_res: Segment size for wind velocity distribution (m/s)
    save: Whether to save the result

    Returns:

    fig: Figure showing the discrete load cases
    u_bins: Edges of wind velocity bins (m)
    h_bins: Edges of water level bins (m)
    data: Discrete load case data

    """
    x_h, h_pdf, x_u, u_pdf = dists
    h_bins = int((max(x_h)-min(x_h))/h_res)
    u_bins = int((max(x_u)-min(x_u))/u_res)
    print('%s h_S-bins by %s u-bins'%(h_bins, u_bins))
    n_u = len(x_u)
    n_h = len(x_h)
    data = []
    for i_h in range(h_bins):
        s_h = slice(round(i_h*n_h/h_bins), round((i_h+1)*n_h/h_bins)+1)
        mean_h = sum(h_pdf[s_h]*x_h[s_h])/sum(h_pdf[s_h])
        for i_u in range(u_bins):
            s_u = slice(round(i_u*n_u/u_bins), round((i_u+1)*n_u/u_bins)+1)
            bin_id = i_h*h_bins + i_u + 1
            prob = np.trapz(u_pdf[s_u], x_u[s_u])*np.trapz(h_pdf[s_h], x_h[s_h])
            mean_u = sum(u_pdf[s_u]*x_u[s_u])/sum(u_pdf[s_u])
            data.append([bin_id, prob, mean_h, mean_u])
    if save:
        with open('../data/03_loadevents/unfiltered_cases.cp.pkl', 'wb') as f:
            cloudpickle.dump(np.array(data), f)
        with open('../data/03_loadevents/currentbins.cp.pkl', 'wb') as bins:
            cloudpickle.dump([x_u, x_h, u_bins, h_bins], bins)
    fig = plt.figure(figsize=(12, 5))
    data = np.array(data)
    hist = plt.hist2d(data[:,2], data[:,3], weights=data[:,1], range=[(min(x_h),max(x_h)),(min(x_u),max(x_u))],
               bins=[h_bins, u_bins], cmap='gist_heat_r', norm=colors.PowerNorm(0.4))
    plt.xlim(x_h[0], x_h[-1])
    plt.ylim(0,40)
    plt.colorbar(hist[3])
    plt.xlabel('d [m]', labelpad=20)
    plt.ylabel('$U_{10}$ [m/s]', labelpad=20)
    plt.close()
    return fig, u_bins, h_bins, np.array(data)

def binfilter(cases, GATE, freqfilter=True, intensityfilter=True):
    """Creates discrete load cases by integrating the probability distributions over small segments.

    Parameters:
    cases: Discrete load case data
    freqfilter: Whether to filter based on probability of occurrence
    intensityfilter: Whether to filter based on maximum expected wave height

    Returns:

    fig: Figure showing the filtered discrete load cases
    df_filt: Filtered discrete load case data

    """
    startcount = len(cases)
    print('Starting with %s cases'%startcount)
    if freqfilter:
        print('Filtering cases with probability less than 1/10,000 years...')
        p = (10000*24*365)**-1
        filtcases = cases[cases[:,1]>p]
        print('%s cases left after frequency filtering'%len(cases))

    if intensityfilter:
        print('Filtering cases with h_S+2*Hm0 < h_G...')
        u_vals = np.flip(np.unique(cases[:,3]))
        h_vals = np.unique(cases[:,2])
        limits = []
        for u_val in u_vals:
            for h_val in h_vals:
                _,_,_,Hm0,_ = spectrum_generator(u_val, h_val)
                if 2*Hm0*(1+cr) + h_val > GATE.HEIGHT:
                    break
            filtcases = np.delete(filtcases, np.where(np.bitwise_and((filtcases[:,2]<h_val),
                                                             (filtcases[:,3]==u_val)))[0], 0)
            h_vals = h_vals[h_vals >= h_val]
        print('%s cases left after Hm0 filtering'%len(filtcases))
    ratio = (startcount-len(filtcases))/startcount*100
    df_filt = pd.DataFrame(data=filtcases[:,1:],
                       index=filtcases[:,0],
                       columns=['p','mean_h','mean_u'])
    print('%d%% of bins filtered out.'%ratio)
    with open('../data/03_loadevents/filtered_cases.cp.pkl', 'wb') as f:
        cloudpickle.dump(df_filt, f)

    with open('../data/03_loadevents/currentbins.cp.pkl', 'rb') as file:
        x_u, x_h, u_bins, h_bins = cloudpickle.load(file)
    df_all  = pd.DataFrame(data=cases[:,1:],
                           index=cases[:,0],
                           columns=['p','mean_h','mean_u'])
    filtered = pd.concat([df_all, df_filt]).drop_duplicates(keep=False)

    fig, ax = plt.subplots()
    load_plot = plt.hist2d(df_filt['mean_h'], df_filt['mean_u'], bins=[h_bins, u_bins],
                           range=[(min(x_h),max(x_h)),(min(x_u),max(x_u))],
                           cmap='gist_heat_r', weights=df_filt['p'],
                           norm=colors.PowerNorm(.3), cmin=1e-10)
    filt_plot = plt.hist2d(filtered['mean_h'], filtered['mean_u'], bins=[h_bins, u_bins],
                           range=[(min(x_h),max(x_h)),(min(x_u),max(x_u))],
                           cmap='gray_r', weights=filtered['p'],
                           norm=colors.PowerNorm(.3), cmin=1e-10)

    plt.colorbar(load_plot[3], format='%.0e', ticks=[1e-6, 1e-5, 1e-4, 1e-3, 5e-3, 1e-2])
    plt.xlabel('$h_S$ [m]', labelpad=20)
    plt.ylabel('$U_{10}$ [m/s]', labelpad=20)
    ax.set_xticks(np.linspace(min(x_h), max(x_h), h_bins+1), minor=True)
    ax.set_yticks(np.linspace(min(x_u), max(x_u), u_bins+1), minor=True)

    plt.xlim(min(df_all['mean_h']), max(df_all['mean_h']))
    plt.ylim(min(df_filt['mean_u']), 45)
    plt.close()
    return fig, df_filt
