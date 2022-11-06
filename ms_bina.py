import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from segment import edge_weight,grad_m
from time import gmtime, strftime
from settings import logger


def msimg(img, ssig=1, rsig=None,init_wt=1):
    """Return mean shift image."""
    fn = strftime("%Y%b%d", gmtime())
    I, J, K = img.shape
    img=img.astype('int')
    pim = np.concatenate((np.zeros((1,J,K)), img, np.zeros((1,J,K))), axis=0) #padding
    pim = np.concatenate((np.zeros((I+2,1,K)), pim,np.zeros((I+2,1,K))), axis=1)
    if init_wt == 1:
        # 1234:ESWN, Io: DHWC, D=8 directions
        grad_E = -pim[1:1 + I, 1:-1, :] + pim[1:1 + I, 2:, :]
        grad_S = -pim[1:-1, 1:1 + J, :] + pim[2:, 1:1 + J, :]
        grad_SE = pim[2:, 2:, :] - pim[1:-1, 1:-1, :]
        grad_NE = pim[:I, 2:, :] - pim[1:-1, 1:-1,:]
        if not rsig:
            rsig = np.mean(np.hstack((grad_E,grad_S,grad_SE,grad_NE)))
        print('rsig %f' % rsig)
        w_E = edge_weight(grad_E,rsig)
        w_E[:,-1]=0
        w_S = edge_weight(grad_S, rsig)
        w_S[-1,:]=0
        w_SE = edge_weight(grad_SE, rsig)
        w_SE[:,-1]=0
        w_SE[-1,:]=0
        w_NE = edge_weight(grad_NE, rsig)
        w_NE[:,-1]=0
        w_NE[0,:]=0
        # fn = 'wt_rsig%f_%s' % (rsig, fn)
        # with open('%s.npy' % fn, 'wb') as f:
        #     np.save(f, np.stack((w_E,w_S,w_SE,w_NE)))

    else:
        fn = 'wt_rsig%f_%s' % (rsig, fn)
        with open('%s.npy' % fn, 'rb') as f:
            wt = np.load(f)
        w_E,w_S,w_SE,w_NE=wt

    wi=np.sum(np.stack((w_E,w_S,w_SE,w_NE)),0)
    si=(np.exp(-wi)>np.random.rand(1)).astype('int')*2-1
    niter=100
    lr=.01
    for i in range(niter):
        m_E=w_E[:,:-1]*si[:,1:]
        m_NE=w_NE[1:,:-1]*si[:-1,1:]
        m_S=w_S[:-1]*si[1:,:]
        m_SE=w_SE[:-1,:-1]*si[1:,1:]
        m_W=w_E[:,:-1]*si[:,:-1]
        m_N=w_S[:-1]*si[:-1,:]
        m_NW=w_SE[:-1,:-1]*si[:-1,:-1]
        m_SW=w_NE[1:,:-1]*si[1:,:-1]
        dwi=np.zeros((I,J))
        dwi[:,:-1]=m_E
        dwi[1:,:-1]+=m_NE
        dwi[:-1,:]+=m_S
        dwi[:-1,:-1]+=m_SE
        dwi[1:,:]+=m_N
        dwi[:,1:]+=m_W
        dwi[1:,1:]+=m_NW
        dwi[:-1,1:]+=m_SW
        wi+=dwi*lr
        si=(np.exp(-wi)>np.random.rand(1)).astype('int')*2-1

    return si


if __name__ == "__main__":

    data_path = os.path.join(os.getcwd(), 'photos')
    im_flist = os.listdir(data_path)
    im_no = 0
    im = mpimg.imread(os.path.join(data_path, im_flist[im_no]))
    # im = im[:200, :200, :]

    ig = msimg(im/1)

    ax = plt.subplot(121)
    # plt.imshow(im[:,:,0])
    plt.imshow(im)
    ax.set_title('sample')
    plt.colorbar(orientation='horizontal')
    ax = plt.subplot(122)
    plt.imshow(ig)
    ax.set_title('one part')
    plt.colorbar(orientation='horizontal')

    plt.show()