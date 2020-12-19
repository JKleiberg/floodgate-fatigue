#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import os
import sys
root_dir = os.path.join(os.getcwd(), '..')
sys.path.append(root_dir)
from src.waveloads.pressure_main import pressure

def worker(args): 
    path, cases = args
    for index, case in cases.iterrows():
        pqs_f, p_imp_t = pressure(Hs= case['mean_Hs'], Tp= case['mean_T'], hsea=case['mean_h']);
        with open(path+'/'+str(int(case['hs_box']))+'_'+str(int(case['h_box']))+'_'+str(int(case['tp_box']))+'.npz', 'wb') as write:
            np.savez_compressed(write, qs_f = pqs_f, imp_t = p_imp_t)  
    return print('saved a file')
