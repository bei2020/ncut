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
        grad_NE = pim[:I, 2:, :] - pim[1:1 + I, 1:-1]
        if not rsig:
            rsig = grad_m(np.hstack((grad_E,grad_S,grad_SE,grad_NE)))
        print('rsig %f' % rsig)
        a = rsig ** 2 / (K + 2)
        r=rsig* 2 ** (1 / nint)
        w_E = edge_weight(grad_E, r)#[I,J]
        w_S = edge_weight(grad_S, r)
        w_SE = edge_weight(grad_SE, r)
        w_NE = edge_weight(grad_NE, r)
        w_W = np.hstack((np.zeros((I, 1)), w_E[:, :-1]))
        w_N = np.vstack((np.zeros((1, J)), w_S[:-1, :,]))
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
        w_E = edge_weight(grad_E,rsig)
        w_S = edge_weight(grad_S,rsig)
        w_SE = edge_weight(grad_SE,rsig)
        w_NE = edge_weight(grad_NE,rsig)
        w_W = np.hstack((np.zeros((I, 1)), w_E[:, :-1]))
        w_N = np.vstack((np.zeros((1, J)), w_S[:-1, :,]))
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
        cur = Iok - Io
        fn = 'wt_rsig%f_%s' % (rsig, fn)
        with open('%s.npy' % fn, 'wb') as f:
            np.save(f, np.hstack((w_E,w_S,w_SE,w_NE)))

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

    def msiter(niter=1000):
        nonlocal pim, v, Vp, w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW, cur
        # wt[:,gloc[0],gloc[1]]=0
        # wt[:,floc[0],floc[1]]=0
        for i in range(niter):
            Io = (np.multiply(w_E.reshape(I,J,1), pim[1:-1, 2:, :]+ gamma * v[1:-1, 2:, :]) + np.multiply(w_S.reshape(I,J,1), pim[2:, 1:-1, :]+ gamma * v[2:, 1:-1, :]) + np.multiply(w_W.reshape(I,J,1), pim[1:-1, :J, :]+ gamma * v[1:-1, :J, :]) \
                  + np.multiply( w_N.reshape(I,J,1), pim[:I, 1:-1, :]+ gamma * v[:I, 1:-1, :]) + np.multiply(w_SE.reshape(I,J,1), pim[2:, 2:, :]+ gamma * v[2:, 2:, :]) + np.multiply(w_NE.reshape(I,J,1), pim[:I, 2:, :]+ gamma * v[:I, 2:, :]) \
                  + np.multiply( w_NW.reshape(I,J,1), pim[:I, :J, :]+ gamma * v[:I, :J, :]) + np.multiply(w_SW.reshape(I,J,1), pim[2:, :J, :]+ gamma * v[2:, :J, :]))/di.reshape((I, J, 1))
            # gp = Io - (img + gamma * v[1:-1,1:-1,:])
            gp = Io - (pim + gamma * v)[1:-1,1:-1,:]
            Vp = alpha * Vp + (1 - alpha) * gp ** 2
            Gp = 1 / (lamb + np.sqrt(Vp))
            v[1:-1,1:-1,:] = gamma * v[1:-1,1:-1,:] + a * (np.multiply(Gp, gp) + cur)
            pim += v
            # c=np.sum(np.multiply(img,di.reshape((I,J,1))),axis=(0,1))
            # img-=c.reshape((1,1,K))/(I*J)/di.reshape((I,J,1))
            # c=np.sum(np.multiply(img**2,di.reshape((I,J,1))),axis=(0,1))
            # img/=(c**(1/2)).reshape((1,1,K))
        return

    def msconti(ncont=0, mcont=12):
        nonlocal pim
        imgc = copy.deepcopy(pim)
        msiter(10)
        if np.array_equal(np.round(np.exp(pim),8),np.round(np.exp(imgc),8)):
        # ginf=1e-12
        # if np.sum(np.square((pim-imgc)[1:-1,1:-1,:]))<ginf:
            print('fix found continue iter %d' % (ncont * 1000))
            print(np.round(pim[:10, :10, 0], 6))
            print(np.round(v[1:-1, 1:-1, 0], 6))
            # print('ground ',pim[gloc[0],gloc[1],:])
            return
        else:
            if ncont == mcont:
                print('no fix after continue iter %d' % (ncont * 1000))
                print(np.round(pim[:10, :10, 0], 6))
                print(np.round(imgc[:10, :10, 0], 6))
                # print('ground ',img[gloc[0],gloc[1],:])
                # print(np.round(np.exp(img[:10,:10,0]), 6))
                # print(np.round(np.exp(imgc[:10,:10,0]), 6))
                return
            print('continue')
            ncont += 1
            msiter(niter=1000)
            msconti(ncont, mcont)
        return

    msiter()
    msconti(mcont=mcont)
    return pim[1:-1,1:-1,:]
    # return img


if __name__ == "__main__":

    # with open('imgs/tri_part','rb') as f:
    #     im=np.load(f)
    # # im=im[8:18,1:11]
    # im=im[4:14,5:15]
    # # iph=im.reshape((*im.shape,1))
    # iph=(im/np.sum(im)).reshape((*im.shape,1))
    # iph=np.sqrt(iph).reshape((*im.shape,1))
    # # iph=np.log(iph).reshape((*im.shape,1))

    data_path = os.path.join(os.getcwd(), 'photos')
    im_flist = os.listdir(data_path)
    im_no = 3
    im = mpimg.imread(os.path.join(data_path, im_flist[im_no]))
    # im=im[110:150,140:190,:]
    im = im[40:60, 10:50, :]
    # im=im[40:80,10:60,:]
    ime = np.einsum('ijk->k', im.astype('uint32')).reshape(1, 1, im.shape[2])
    iph = im / ime
    iph=np.sqrt(iph)

    I, J, K = iph.shape
    l = np.arange(I * J).reshape(I, J)
    ig = msimg(copy.deepcopy(iph), mcont=1)
    # ig=msimg(copy.deepcopy(iph),rsig=.1,mcont=4)
    # ig=msimg(ig,rsig=.08)

    # blabels=np.zeros((I,J,6))
    # prob = 1 / (1 + np.exp(-np.sum(np.multiply(ig,iph),axis=2)))
    # prob = 1 / (1 + np.exp(-ig[:,:,ch]))
    # blabels[:,:,0] = (prob > np.random.rand(*prob.shape)).astype('int')

    ax = plt.subplot(131)
    # plt.imshow(im[:,:,0])
    plt.imshow(im)
    ax.set_title('sample')
    plt.colorbar(orientation='horizontal')
    ax = plt.subplot(132)
    # plt.imshow(np.square(ig[:,:,0])*np.sum(im))
    plt.imshow(ig[:, :, 0])
    ax.set_title('ig')
    plt.colorbar(orientation='horizontal')
    # ax=plt.subplot(133)
    # plt.imshow(blabels[:,:,0])
    # ax.set_title('seg1 4000')
    # plt.colorbar(orientation='horizontal')

    ax = plt.subplot(133)
    plt.imshow(iph[:, :, 0])
    ax.set_title('iph')
    plt.colorbar(orientation='horizontal')

    plt.show()