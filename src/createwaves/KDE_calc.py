#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import multiprocessing
import numpy as np
import cloudpickle
import os
import sys
root_dir = os.path.join(os.getcwd(), '..')
sys.path.append(root_dir)

def calc_kernel(samp):
    with open('../data/03_intermediate/test.cp.pkl', 'rb') as f:
        kernel = cloudpickle.load(f)
    return kernel(samp)

# Create definition.
def KDE_calc(xyz):  
    #Choose number of cores and split input array.
    cores = multiprocessing.cpu_count()
    torun = np.array_split(xyz, cores, axis=1)

    # #Calculate
    pool = multiprocessing.Pool(processes=cores)
    results = pool.map(calc_kernel, torun)

    #Reintegrate and calculate results
    density = np.concatenate(results)
    return density

