#!/usr/bin/env python
# coding: utf-8
def create_waves(data,rise,plot=False):
    import random 
    import numpy as np
    import pandas as pd
    from src.configuration import g, h0, fetch

    data = pd.read_csv(data, index_col=0)
    data.dropna(inplace=True)
    d = data['h']
    u10 = data['wind']
    hinf = 0.24
    tinf = 7.69
    Hs = []
    Tp = []
    dss= []
    for i in range(len(d)):
        dss.append(d[i] + h0 + random.uniform(0,rise))
        facc = g*fetch/u10[i]**2
        dacc = g*dss[i]/u10[i]**2
        hacc = hinf*(np.tanh(0.343*dacc**1.14)*np.tanh(4.41*10**-4*facc**0.79/np.tanh(0.343*dacc**1.14)))**0.572
        Hs.append(hacc*u10[i]**2/g)
        tacc = tinf*(np.tanh(0.1*dacc**2.01)*np.tanh(2.77*10**-7*facc**1.45/np.tanh(0.1*dacc**2.01)))**0.187
        Tp.append(tacc*u10[i]/g)
    data['Hs'] = Hs
    data['Tp'] = Tp
    data['h'] = dss
    data.to_csv(r'../data/02_preprocessing/wavedata.csv')
    