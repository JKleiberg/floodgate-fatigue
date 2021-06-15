import dill
import numpy as np
# Load system properties
with open('../data/06_transferfunctions/current_case.pkl', 'rb') as file:
    GATE = dill.load(file)
def nearest(x,y,z):
    coord = (x,y,z)
    array = np.asarray(GATE.coords)
    sq_dist = (array[:,0]-coord[0])**2 + (array[:,1]-coord[1])**2 + (array[:,2]-coord[2])**2
    idx = sq_dist.argmin()
    return tuple(array[idx])
