import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from segment import edge_weight,grad_m
import copy
import os
from time import gmtime, strftime


def msimg(img, rsig=None, nlable=3, init_wt=1):
    """Return mean shift image."""
    fn = strftime("%Y%b%d", gmtime())
    I, J, K = img.shape
    img=img.astype('int')
    pim = np.concatenate((np.zeros((1,J,K)), img, np.zeros((1,J,K))), axis=0) #padding
    pim = np.concatenate((np.zeros((I+2,1,K)), pim,np.zeros((I+2,1,K))), axis=1)
    if init_wt == 1:
        grad_E = pim[1:1 + I, 2:, :]-pim[1:1 + I, 1:-1, :]
        grad_S = pim[2:, 1:1 + J, :]-pim[1:-1, 1:1 + J, :]
        grad_SE = pim[2:, 2:, :] - pim[1:-1, 1:-1, :]
        grad_NE = pim[:I, 2:, :] - pim[1:-1, 1:-1,:]
        if not rsig:
            rsig = grad_m(np.hstack((abs(grad_E),abs(grad_S),abs(grad_SE),abs(grad_NE))))
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
        # wn=2*np.sum([w_E,w_S,w_SE,w_NE])
        # w_E/=wn
        # w_S/=wn
        # w_SE/=wn
        # w_NE/=wn
        # fn = 'wt_rsig%f_%s' % (rsig, fn)
        # with open('%s.npy' % fn, 'wb') as f:
        #     np.save(f, np.stack((w_E,w_S,w_SE,w_NE)))

    else:
        fn = 'wt_rsig%f_%s' % (rsig, fn)
        with open('%s.npy' % fn, 'rb') as f:
            wt = np.load(f)
        w_E,w_S,w_SE,w_NE=wt

    q=np.random.rand(I,J,nlable)
    q/=np.sum(q,-1).reshape(I,J,1)
    pls=-np.ones((nlable,nlable))+2*np.diag(np.ones(nlable))
    niter=100
    T=round(np.exp(-1/2)/(4*np.log(1+np.sqrt(nlable))),2)
    for i in range(niter):
        q = np.dot(q, pls)
        m_E=w_E[:,:-1].reshape(I,J-1,1)*q[:,1:,:]
        m_NE=w_NE[1:,:-1].reshape(I-1,J-1,1)*q[:-1,1:,:]
        m_S=w_S[:-1,:].reshape(I-1,J,1)*q[1:,:,:]
        m_SE=w_SE[:-1,:-1].reshape(I-1,J-1,1)*q[1:,1:,:]
        m_W=w_E[:,:-1].reshape(I,J-1,1)*q[:,:-1,:]
        m_N=w_S[:-1,:].reshape(I-1,J,1)*q[:-1,:,:]
        m_NW=w_SE[:-1,:-1].reshape(I-1,J-1,1)*q[:-1,:-1,:]
        m_SW=w_NE[1:,:-1].reshape(I-1,J-1,1)*q[1:,:-1,:]
        q=np.zeros((I,J,nlable))
        q[:,:-1,:]=m_E
        q[1:,:-1,:]+=m_NE
        q[:-1,:,:]+=m_S
        q[:-1,:-1,:]+=m_SE
        q[1:,:,:]+=m_N
        q[:,1:,:]+=m_W
        q[1:,1:,:]+=m_NW
        q[:-1,1:,:]+=m_SW
        q=np.exp(q/T)
        q/=np.sum(q,-1).reshape(I,J,1)

    return q
    # return img


if __name__ == "__main__":
    #E,S,W,N,SE,SW,NW,NE
    lij=((0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,-1),(-1,1))

    data_path = os.path.join(os.getcwd(), 'photos')
    im_flist = os.listdir(data_path)
    # im_no = 0
    im_no = -1
    im = mpimg.imread(os.path.join(data_path, im_flist[im_no]))
    # im=im[110:150,140:190,:]
    # im= im[400:410, 610:625, :]
    # im= im[400:440, 610:650, :]
    im=im[40:80,10:60,:]
    # im = im[40:60, 10:40, :]
    iph=im/1

    I, J, K = iph.shape
    ig = msimg(iph,nlable=3)
    b1=(np.random.rand()<ig[:,:,0]).astype('int')

    ax = plt.subplot(131)
    # plt.imshow(im[:,:,0])
    plt.imshow(im)
    ax.set_title('sample')
    plt.colorbar(orientation='horizontal')
    ax = plt.subplot(132)
    plt.imshow(ig)
    ax.set_title('ig')
    # plt.imshow(np.square(ig[:,:,0])*np.sum(im))
    plt.colorbar(orientation='horizontal')
    # ax=plt.subplot(133)
    # plt.imshow(blabels[:,:,0])
    # ax.set_title('seg1 4000')
    # plt.colorbar(orientation='horizontal')

    ax = plt.subplot(133)
    plt.imshow(ig[:, :, 0])
    ax.set_title('cluster')
    plt.colorbar(orientation='horizontal')

    # ax = plt.subplot(144)
    # plt.imshow((ig[:,:,0]<0).astype('int'))
    # ax.set_title('seg0')
    # plt.colorbar(orientation='horizontal')

    plt.show()