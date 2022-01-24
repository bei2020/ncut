import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from segment import edge_weight,grad_m
import copy
import os
from time import gmtime, strftime


def msimg(img, ssig=1, rsig=None, mcont=5, init_wt=1):
    """Return mean shift image."""
    fn = strftime("%Y%b%d", gmtime())
    I, J, K = img.shape
    img=img.astype('int')
    pim = np.concatenate((np.zeros((1,J,K)), img, np.zeros((1,J,K))), axis=0) #padding
    pim = np.concatenate((np.zeros((I+2,1,K)), pim,np.zeros((I+2,1,K))), axis=1)
    gamma = .9
    if init_wt == 1:
        grad_E = pim[1:1 + I, 2:, :]-pim[1:1 + I, 1:-1, :]
        grad_S = pim[2:, 1:1 + J, :]-pim[1:-1, 1:1 + J, :]
        grad_SE = pim[2:, 2:, :] - pim[1:-1, 1:-1, :]
        grad_NE = pim[:I, 2:, :] - pim[1:-1, 1:-1,:]
        if not rsig:
            rsig = grad_m(np.hstack((grad_E,grad_S,grad_SE,grad_NE)))
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
        wn=2*np.sum([w_E,w_S,w_SE,w_NE])
        w_E/=wn
        w_S/=wn
        w_SE/=wn
        w_NE/=wn
        # fn = 'wt_rsig%f_%s' % (rsig, fn)
        # with open('%s.npy' % fn, 'wb') as f:
        #     np.save(f, np.stack((w_E,w_S,w_SE,w_NE)))

    else:
        fn = 'wt_rsig%f_%s' % (rsig, fn)
        with open('%s.npy' % fn, 'rb') as f:
            wt = np.load(f)
        w_E,w_S,w_SE,w_NE=wt
    a=.001


    def msiter(pim, w_E,w_S,w_SE,w_NE,niter=100):
        v = np.zeros((I+2,J+2,K))
        for i in range(niter):
            q_E = np.exp(-np.multiply(w_E.reshape(I,J,1), (pim[1:-1, 2:, :]-pim[1:-1, 1:-1, :] + gamma * v[1:-1, 2:, :]-gamma * v[1:-1,1:-1,:])**2))
            q_S = np.exp(-np.multiply(w_S.reshape(I,J,1), (pim[2:, 1:-1, :]-pim[1:-1, 1:-1, :] + gamma * v[2:, 1:-1, :]-gamma * v[1:-1,1:-1,:])**2))
            q_SE = np.exp(-np.multiply(w_SE.reshape(I,J,1),(pim[2:, 2:, :] - pim[1:-1, 1:-1, :]+ gamma * v[2:, 2:, :]-gamma * v[1:-1,1:-1,:])**2))
            q_NE = np.exp(-np.multiply(w_NE.reshape(I,J,1),(pim[:-2, 2:, :] - pim[1:-1, 1:-1,:]+ gamma * v[:-2, 2:, :]-gamma * v[1:-1,1:-1,:])**2))
            q_E[:,-1]=0
            q_S[-1,:]=0
            q_SE[:,-1]=0
            q_SE[-1,:]=0
            q_NE[:,-1]=0
            q_NE[0,:]=0
            Io=q_E*pim[1:-1, 2:, :]+q_S*pim[2:, 1:-1, :]+q_SE*pim[2:, 2:, :]+q_NE*pim[:-2, 2:, :]
            Io[:,1:,:]+=q_E[:,:-1,:]*pim[1:-1, 1:-2, :]
            Io[1:,:,:]+=q_S[:-1,:,:]*pim[1:-2, 1:-1, :]
            Io[1:,1:,:]+=q_SE[:-1,:-1,:]*pim[1:-2, 1:-2, :]
            Io[:-1,1:,:]+=q_NE[1:,:-1,:]*pim[2:-1, 1:-2, :]
            di=q_E+q_S+q_SE+q_NE
            di[:,1:,:]+=q_E[:,:-1,:]
            di[1:,:,:]+=q_S[:-1,:,:]
            di[1:,1:,:]+=q_SE[:-1,:-1,:]
            di[:-1,1:,:]+=q_NE[1:,:-1,:]
            Io=Io/di-pim[1:-1, 1:-1, :]
            v[1:-1,1:-1,:] = gamma * v[1:-1,1:-1,:] + a * Io
            pim += v
        return

    msiter(pim, w_E,w_S,w_SE,w_NE)
    return pim[1:-1,1:-1,:]
    # return img


if __name__ == "__main__":
    #E,S,W,N,SE,SW,NW,NE
    lij=((0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,-1),(-1,1))

    data_path = os.path.join(os.getcwd(), 'photos')
    im_flist = os.listdir(data_path)
    # im_no = 1
    im_no = 4
    im = mpimg.imread(os.path.join(data_path, im_flist[im_no]))
    # im=im[110:150,140:190,:]
    # im= im[400:410, 610:625, :]
    # im= im[400:440, 610:650, :]
    # im=im[40:80,10:60,:]
    im = im[40:60, 10:50, :]
    iph=im/1

    I, J, K = iph.shape
    ig = msimg(iph)

    ax = plt.subplot(131)
    # plt.imshow(im[:,:,0])
    plt.imshow(im)
    ax.set_title('sample')
    plt.colorbar(orientation='horizontal')
    ax = plt.subplot(132)
    plt.imshow(iph[:, :, 0])
    ax.set_title('iph')
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