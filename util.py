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

def mpb_maxloc(img):
    I,J,K=img.shape
    sig=2
    ks=2*sig+1
    ni=I//ks
    nj=J//ks
    cloc=lambda i,j:(ks*i+sig,ks*j+sig)
    G=np.zeros((ni,nj))
    ch=0
    for i in range(ni):
        for j in range(nj):
            ci,cj=cloc(i,j)
            #0
            g=np.histogram(img[ci-sig:ci,cj-sig:cj+sig+1,ch],bins=30,range=(0,255))[0]
            h=np.histogram(img[ci+1:ci+sig+1,cj-sig:cj+sig+1,ch],bins=30,range=(0,255))[0]
            G[i,j]+=np.sum((g-h)**2)
    gj=np.argmax(G)
    gi=gj//nj
    gj=gj%nj
    return ks*gi+sig,ks*gj+sig

if __name__ == "__main__":
    data_path = os.path.join(os.getcwd(), 'photos')
    im_flist = os.listdir(data_path)
    im_no = 0
    im = mpimg.imread(os.path.join(data_path, im_flist[im_no]))
    # im = im[40:60, 10:40, :]
    im=im[110:140,160:180,:]
    gloc=mpb_maxloc(im)
    print(gloc)
