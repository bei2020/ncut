import numpy as np
from matplotlib import pyplot as plt
import os
import matplotlib.image as mpimg

def conv2d_ch1(img,w_E, w_S, w_SE, w_NE, w_W, w_N, w_NW, w_SW,w_i=-1):
    """Return 2d spatial convolved 1channel img.
    """
    I,J=img.shape
    pim=np.zeros((I+2,J+2))
    pim[1:-1,1:-1]=img
    pim[0,1:-1]=img[0,:]
    pim[-1,1:-1]=img[-1,:]
    pim[:,0]=pim[1,1]
    pim[:,-1]=pim[:,-2]
    return w_E*pim[1:-1,2:]+w_S*pim[2:,1:-1]+w_W*pim[1:-1,:J]+w_N*pim[:I,1:-1]+w_SE*pim[2:,2:]+w_NE*pim[:I,2:] \
           + w_NW*pim[:I,:J]+w_SW*pim[2:,:J]+w_i*img

