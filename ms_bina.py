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
    pim = np.concatenate((np.zeros((1,J,K)), img, np.zeros((1,J,K))), axis=0) #padding
    pim = np.concatenate((np.zeros((I+2,1,K)), pim,np.zeros((I+2,1,K))), axis=1)
    gamma = .9
    alpha = .99
    lamb = 1e-5
    v = np.zeros((I+2,J+2,K))
    Vp = 0
    if init_wt == 1:
        nint = 2
        # 1234:ESWN, Io: DHWC, D=8 directions
        # curvature
        grad_E = -pim[1:1 + I, 1:-1, :] + pim[1:1 + I, 2:, :]
        grad_S = -pim[1:-1, 1:1 + J, :] + pim[2:, 1:1 + J, :]
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
        w_W = np.hstack((np.zeros((I, 1)), w_E[:, :-1]))
        w_N = np.vstack((np.zeros((1, J)), w_S[:-1, :]))
        w_NW = np.hstack((np.zeros((I, 1)),np.vstack((np.zeros((1, J - 1)), w_SE[:-1,:-1]))))
        w_SW = np.hstack((np.zeros((I, 1)), np.vstack((w_NE[1:,:-1], np.zeros((1, J - 1))))))
        wn=np.sum([w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW])
        w_E/=wn
        w_S/=wn
        w_SE/=wn
        w_NE/=wn
        w_W/=wn
        w_N/=wn
        w_NW/=wn
        w_SW/=wn
        di=np.sum(np.stack((w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW),0),0)
        gloc = np.argmin(di[1:-1, 1:-1])
        gloc = (gloc // (J - 2) + 1, gloc % (J - 2) + 1)
        print('gloc %d %d' % (gloc[0], gloc[1]))
        floc = np.argmin(np.stack((w_E, w_S,w_W,w_N,w_SE, w_SW, w_NW,w_NE))[:, gloc[0], gloc[1]])
        floc = (gloc[0] + lij[floc][0], gloc[1] + lij[floc][1])
        print('floc %d %d' % (floc[0], floc[1]))
        rsig=np.max(abs(img[gloc]-im[floc]))
        print('rsig gf %f' % rsig)

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
        w_W = np.hstack((np.zeros((I, 1)), w_E[:, :-1]))
        w_N = np.vstack((np.zeros((1, J)), w_S[:-1, :]))
        w_NW = np.hstack((np.zeros((I, 1)),np.vstack((np.zeros((1, J - 1)), w_SE[:-1,:-1]))))
        w_SW = np.hstack((np.zeros((I, 1)), np.vstack((w_NE[1:,:-1], np.zeros((1, J - 1))))))
        wn=np.sum([w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW])
        w_E/=wn
        w_S/=wn
        w_SE/=wn
        w_NE/=wn
        w_W/=wn
        w_N/=wn
        w_NW/=wn
        w_SW/=wn
        di=np.sum(np.stack((w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW),0),0)
        Io = (np.multiply(w_E.reshape(I,J,1), pim[1:-1, 2:, :]) + np.multiply(w_S.reshape(I,J,1), pim[2:, 1:-1, :]) + np.multiply(w_W.reshape(I,J,1), pim[1:-1, :J, :]) \
              + np.multiply( w_N.reshape(I,J,1), pim[:I, 1:-1, :]) + np.multiply(w_SE.reshape(I,J,1), pim[2:, 2:, :]) + np.multiply(w_NE.reshape(I,J,1), pim[:I, 2:, :]) \
              + np.multiply( w_NW.reshape(I,J,1), pim[:I, :J, :]) + np.multiply(w_SW.reshape(I,J,1), pim[2:, :J, :]))/di.reshape((I, J, 1))

        rsig=rsig* 2 ** (1 / nint)
        w_E = edge_weight(grad_E, rsig)#[I,J]
        w_E[:,-1]=0
        w_S = edge_weight(grad_S, rsig)
        w_S[-1,:]=0
        w_SE = edge_weight(grad_SE, rsig)
        w_SE[:,-1]=0
        w_SE[-1,:]=0
        w_NE = edge_weight(grad_NE, rsig)
        w_NE[:,-1]=0
        w_NE[0,:]=0
        w_W = np.hstack((np.zeros((I, 1)), w_E[:, :-1]))
        w_N = np.vstack((np.zeros((1, J)), w_S[:-1, :]))
        w_NW = np.hstack((np.zeros((I, 1)),np.vstack((np.zeros((1, J - 1)), w_SE[:-1,:-1]))))
        w_SW = np.hstack((np.zeros((I, 1)), np.vstack((w_NE[1:,:-1], np.zeros((1, J - 1))))))
        wn=np.sum([w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW])
        w_E/=wn
        w_S/=wn
        w_SE/=wn
        w_NE/=wn
        w_W/=wn
        w_N/=wn
        w_NW/=wn
        w_SW/=wn
        di=np.sum(np.stack((w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW),0),0)
        Iok = (np.multiply(w_E.reshape(I,J,1), pim[1:-1, 2:, :]) + np.multiply(w_S.reshape(I,J,1), pim[2:, 1:-1, :]) + np.multiply(w_W.reshape(I,J,1), pim[1:-1, :J, :])\
             + np.multiply( w_N.reshape(I,J,1), pim[:I, 1:-1, :]) + np.multiply(w_SE.reshape(I,J,1), pim[2:, 2:, :]) + np.multiply(w_NE.reshape(I,J,1), pim[:I, 2:, :]) \
             + np.multiply( w_NW.reshape(I,J,1), pim[:I, :J, :]) + np.multiply(w_SW.reshape(I,J,1), pim[2:, :J, :]))/di.reshape((I, J, 1))
        cur = Iok - Io

    else:
        fn = 'wt_rsig%f_%s' % (rsig, fn)
        a = rsig ** 2 / (K + 2)
        with open('%s.npy' % fn, 'rb') as f:
            wt = np.load(f)
        w_E,w_S,w_SE,w_NE=wt
        w_W = np.hstack((np.zeros((I, 1)), w_E[:, :-1]))
        w_N = np.vstack((np.zeros((1, J)), w_S[:-1, :,]))
        w_NW = np.hstack((np.zeros((I, 1)),np.vstack((np.zeros((1, J - 1)), w_SE[:-1,:-1]))))
        w_SW = np.hstack((np.zeros((I, 1)), np.vstack((w_NE[1:,:-1], np.zeros((1, J - 1))))))

    return cur
    # return img


if __name__ == "__main__":
    # E,S,W,N,SE,SW,NW,NE
    lij=((0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,-1),(-1,1))

    data_path = os.path.join(os.getcwd(), 'photos')
    im_flist = os.listdir(data_path)
    # im_no = 5
    im_no = 1
    # im_no = 3
    im = mpimg.imread(os.path.join(data_path, im_flist[im_no]))
    # im=im[110:150,140:190,:]
    im=im[100:300,:300,:]
    # im= im[400:410, 610:625, :]
    # im= im[400:440, 610:650, :]
    # im = im[40:60, 10:50, :]
    # im=im[40:80,10:60,:]
    # im=im[45:55,50:60,:]
    # im = im[40:60, 10:50, :]
    # im= im[360:400, 450:650, :]
    iph=im/1

    I, J, K = iph.shape
    ig = msimg(copy.deepcopy(iph), mcont=0)

    h, b = np.histogram(abs(ig).flatten(), bins=20)
    th=np.dot(h, b[1:]) / np.sum(h)

    ax = plt.subplot(141)
    # plt.imshow(im[:,:,0])
    plt.imshow(im)
    ax.set_title('sample')
    plt.colorbar(orientation='horizontal')
    ax = plt.subplot(142)
    plt.imshow(ig[:, :, 0])
    ax.set_title('cur')
    # plt.imshow(np.square(ig[:,:,0])*np.sum(im))
    plt.colorbar(orientation='horizontal')
    # ax=plt.subplot(133)
    # plt.imshow(blabels[:,:,0])
    # ax.set_title('seg1 4000')
    # plt.colorbar(orientation='horizontal')

    ax = plt.subplot(143)
    plt.imshow((ig[:, :, 0]>th).astype('int'))
    ax.set_title('edge')
    plt.colorbar(orientation='horizontal')

    ax = plt.subplot(144)
    plt.imshow((ig[:,:,0]>0).astype('int'))
    ax.set_title('cur>0')
    plt.colorbar(orientation='horizontal')

    plt.show()