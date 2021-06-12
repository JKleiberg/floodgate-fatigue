import numpy as np
from src.configuration import Gate
def nearest(x,y,z):
    coord = (x,y,z)
    array = np.asarray(Gate.verts)
    sq_dist = (array[:,0]-coord[0])**2 + (array[:,1]-coord[1])**2 + (array[:,2]-coord[2])**2
    idx = sq_dist.argmin()
    return tuple(array[idx])
