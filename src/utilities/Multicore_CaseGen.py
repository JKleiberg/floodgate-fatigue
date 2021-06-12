#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import os
import sys
root_dir = os.path.join(os.getcwd(), '..')
sys.path.append(root_dir)
from src.pressure import pressure

def worker(args): 
    path, cases = args
    for index, case in cases.iterrows():
        freqs, pqs_f, impact_force_t,Hs,Tp = pressure(u_wind=case['mean_U'], hsea=case['mean_d']);
        with open(path+'/'+str(int(case['bin']))+'.npz', 'wb') as write:
            np.savez_compressed(write, qs_f=pqs_f, imp_t=impact_force_t)  
    return

from src.configuration import cr, H_GATE
from src.spec import spectrum_generator

# def filterworker(cases):
#     indices = []
#     for i in range(len(cases)):
#         f,amp,k,Hm0,Tp = spectrum_generator(cases['mean_U'].iloc[i],cases['mean_d'].iloc[i])
#         if 2*Hm0*(1+cr) + cases['mean_d'].iloc[i] < H_GATE:
#             indices.append(cases.index[i])
#     return indices

def filterworker(cases):
    indices = []
    for i, case in enumerate(cases):
        bin_id, prob, mean_h, mean_u = case
        f,amp,k,Hm0,Tp = spectrum_generator(mean_u, mean_h)
        if 2*Hm0*(1+cr) + mean_h > H_GATE:
            indices.append(int(bin_id))
    return indices
