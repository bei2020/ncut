"""learn coefficients of SG least square convolution."""
import numpy as np
from matplotlib import pyplot as plt
from settings import logger, ZEROTH

nV=2 # value levels

def highest_power2(n):
    """Return largest power of 2 lower than n"""
    for i in range(n,1,-1):
        if (i&(i-1))==0:
            return i

def wt_LS(V):
    """Return weight."""
    inte=1/(nV-1)
    wc=[i*inte for  i in range(nV)]
    addr=np.argmin(V)
    hp = highest_power2(addr)
    pw = np.log2(hp).astype('int')
    depth = pw + 1 if addr > hp else pw
    wtv=[addr%nV**j//nV**(j-1) for j in range(depth,0,-1)]
    # wt=[wc[i] for i in wtv]
    return [wc[i] for i in wtv]

def value_wt(v,img,V):
    """Return Value of weight.
    v:w_E w_S w_SE w_NE 4 elements wt level index vector,total c levels
    V: Value function,LUT(address(v))"""
    addr=address(v)
    return V

def address(v):
    """Return sumj(v_j*c^j-1)"""
    addr = 0
    for ve in v:
        addr *= nV
        addr += ve
    return addr

def conv2d(img,wt):
    """Return 2d spatial convolved img."""
    I, J, K = img.shape
    pim = np.concatenate((np.zeros((1, J, K)), img, np.zeros((1, J, K))), axis=0)  # padding
    pim = np.concatenate((np.zeros((I + 2, 1, K)), pim, np.zeros((I + 2, 1, K))), axis=1)
    return img


if __name__ == "__main__":

    with open('imgs/tri_part', 'rb') as f:
        im = np.load(f)
    # im=im[4:14,5:15]
    iph = (im / np.sum(im))

    # data_path = os.path.join(os.getcwd(), 'photos')
    # im_flist = os.listdir(data_path)
    # im_no = 2
    # im = mpimg.imread(os.path.join(data_path, im_flist[im_no]))
    # # im=im[110:150,140:190,:]
    # # im = im[180:280, 100:150, :]
    # im= im[400:440, 610:650, :]
    # ime = np.einsum('ijk->k', im.astype('uint32')).reshape(1, 1, im.shape[2])
    # iph = im / ime
    # iph[iph == 0] = .0000001 / np.sum(ime)
    # iph=np.sqrt(iph)

    v=[1,0,1,0]
    V=[0.0]*nV**4
    V=value_wt(v,iph,V)
    wt=wt_LS(V)

    imc = conv2d(iph,wt)

    ax = plt.subplot(231)
    # plt.imshow(im[:,:,0])
    plt.imshow(im)
    ax.set_title('sample')
    plt.colorbar(orientation='horizontal')
    ax = plt.subplot(232)
    # plt.imshow(np.square(ig[:,:,0])*np.sum(im))
    plt.imshow(ig[:, :, 0])
    ax.set_title('ig')
    plt.colorbar(orientation='horizontal')
    # ax=plt.subplot(233)
    # plt.imshow(blabels[:,:,0])
    # ax.set_title('seg1 4000')
    # plt.colorbar(orientation='horizontal')

    ax = plt.subplot(233)
    plt.imshow(iph[:, :, 0])
    ax.set_title('iph')
    plt.colorbar(orientation='horizontal')


    plt.show()
