"""learn coefficients of SG least square convolution."""
import numpy as np
from matplotlib import pyplot as plt
from settings import logger,INIT_V,SAVE_WT
import matplotlib.image as mpimg
import os


def v2wt(v):
    w=[wc[i] for i in v]
    if np.sum(w)==0:
        return np.zeros(5)
    else:
        return w/(2*np.sum(w))

def wt_LS(V):
    """Return weight."""
    addr=np.argmin(V)
    v=[]
    for i in range(5):
        v.append(addr%nV)
        addr=addr//nV
    v=[v[4-i] for i in range(5)]#reverse
    logger.info('bestv %s'%str(v))
    return v2wt(v)

def address(v):
    """Return sumj(v_j*c^j-1)"""
    addr = 0
    for ve in v:
        addr *= nV
        addr += ve
    return addr

def conv2d_ch1(img,w_E, w_S, w_SE, w_NE, w_W, w_N, w_NW, w_SW,w_i):
    """Return 2d spatial convolved img."""
    I,J=img.shape
    pim=np.zeros((I+2,J+2))
    pim[1:-1,1:-1]=img
    pim[0,1:-1]=img[0,:]
    pim[-1,1:-1]=img[-1,:]
    pim[:,0]=pim[1,1]
    pim[:,-1]=pim[:,-2]
    return w_E*pim[1:-1,2:]+w_S*pim[2:,1:-1]+w_W*pim[1:-1,:J]+w_N*pim[:I,1:-1]+w_SE*pim[2:,2:]+w_NE*pim[:I,2:] \
           + w_NW*pim[:I,:J]+w_SW*pim[2:,:J]+w_i*img

def value_update_ch1(v,img,V):
    """Return updated Value of weight.
    v:w_E w_S w_SE w_NE 4 elements wt level index vector,total c levels
    V: Value function,LUT(address(v))"""
    I,J=img.shape
    alpha=.01
    addr=address(v)
    wt=v2wt(v)
    ms2=(conv2d_ch1(img,*wt[:-1],*wt)-img)**2
    for i in range(I-2):
        for j in range(J-2):
            V[addr]=V[addr]+alpha*(np.sum(ms2[i:i+3,j:j+3])-V[addr])
    return V

def Vupdate(img,V):
    # v=[0,0,1,0,1]
    for i in range(nV):
        for j in range(nV):
            for k in range(nV):
                for l in range(nV):
                    for m in range(nV):
                        v=[i,j,k,l,m]
                        V=value_update_ch1(v,img,V)
                        V=value_update_ch1(v,np.rot90(img),V)
                        imgm=np.fliplr(img)
                        V=value_update_ch1(v,imgm,V)
                        V=value_update_ch1(v,np.rot90(imgm),V)
    return V

if __name__ == "__main__":
    nV=3 # value levels=c
    # wc: [-1,1]
    inte=2/(nV-1)
    wc=[i*inte-1 for  i in range(nV)]


    fn = 'Vwt_SGLS_nV%d.npy'%nV
    if INIT_V==1:
        with open('imgs/tri_part', 'rb') as f:
            im = np.load(f)
        iph = (im / np.sum(im))
        V=[1]*nV**5
        for i in range(nV):
            for j in range(nV):
                for k in range(nV):
                    for l in range(nV):
                        for m in range(nV):
                            v=[i,j,k,l,m]
                            wt=v2wt(v)
                            V[address(v)]=np.sum((conv2d_ch1(iph[:3,:3],*wt[:-1],*wt)-iph[:3,:3])**2)
    else:
        with open( fn, 'rb') as f:
            V = np.load(f)
        logger.info('--- loaded weights value')

    data_path = os.path.join(os.getcwd(), 'photos')
    im_flist = os.listdir(data_path)
    for imn in im_flist[3:-1]:
        im = mpimg.imread(os.path.join(data_path, imn))
        ime = np.einsum('ijk->k', im.astype('uint32')).reshape(1, 1, 3)
        iph = im / ime
        for k in range(3):
            V=Vupdate(iph[:20,:30,:][:,:,k],V)

    if SAVE_WT:
        with open( fn, 'wb') as f:
            np.save(f, V)
    wt=wt_LS(V)
    logger.info('bestwt %s'%str(wt))

    im = mpimg.imread(os.path.join(data_path, im_flist[3]))
    ime = np.einsum('ijk->k', im.astype('uint32')).reshape(1, 1, 3)
    iph = im / ime
    imc = [conv2d_ch1(iph[:,:,k],*wt[:-1],*wt) for k in range(3)]
    imc=np.transpose(np.array(imc),(1,2,0))

    ax = plt.subplot(131)
    plt.imshow(im)
    ax.set_title('sample')
    plt.colorbar(orientation='horizontal')
    ax = plt.subplot(132)
    # plt.imshow(imc)
    plt.imshow((imc*ime).astype('int'))
    ax.set_title('imc')
    plt.colorbar(orientation='horizontal')

    ax = plt.subplot(133)
    plt.imshow(imc)
    plt.imshow((iph*ime).astype('int'))
    ax.set_title('iph')
    plt.colorbar(orientation='horizontal')

    plt.show()
