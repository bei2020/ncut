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
    v = np.zeros((I+2,J+2,K))
    if init_wt == 1:
        nint = 2
        # 1234:ESWN, Io: DHWC, D=8 directions
        grad_E = pim[1:1 + I, 2:, :]-pim[1:1 + I, 1:-1, :]
        grad_S = pim[2:, 1:1 + J, :]-pim[1:-1, 1:1 + J, :]
        grad_SE = pim[2:, 2:, :] - pim[1:-1, 1:-1, :]
        grad_NE = pim[:I, 2:, :] - pim[1:-1, 1:-1,:]
        if not rsig:
            rsig = grad_m(np.hstack((grad_E,grad_S,grad_SE,grad_NE)))
        print('rsig %f' % rsig)
        a = rsig ** 2 / (K + 2)
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
        a = rsig ** 2 / (K + 2)
        with open('%s.npy' % fn, 'rb') as f:
            wt = np.load(f)
        w_E,w_S,w_SE,w_NE=wt

    w_W = np.hstack((np.zeros((I, 1)), w_E[:, :-1]))
    w_N = np.vstack((np.zeros((1, J)), w_S[:-1, :,]))
    w_NW = np.hstack((np.zeros((I, 1)),np.vstack((np.zeros((1, J - 1)), w_SE[:-1,:-1]))))
    w_SW = np.hstack((np.zeros((I, 1)), np.vstack((w_NE[1:,:-1], np.zeros((1, J - 1))))))
    di=np.sum(np.stack((w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW),0),0)
    gloc=np.argmin(di[1:-1,1:-1])
    gloc = (gloc // (J - 2) + 1, gloc % (J - 2) + 1)
    print('gloc %d %d'%(gloc[0],gloc[1]))
    fid=np.argmin(np.stack((w_E,w_S,w_W,w_N,w_SE,w_SW,w_NW,w_NE))[:,gloc[0],gloc[1]])
    floc=(gloc[0]+lij[fid][0],gloc[1]+lij[fid][1])
    [w_E,w_S,w_W,w_N,w_SE,w_SW,w_NW,w_NE][fid][gloc[0],gloc[1]]=0
    [w_W,w_N,w_E,w_S,w_NW,w_NE,w_SE,w_SW][fid][floc[0],floc[1]]=0
    print('floc %d %d'%(floc[0],floc[1]))

    pim[floc[0]+1,floc[1]+1,:]+=pim[gloc[0]+1,gloc[1]+1,:]
    pim[gloc[0]+1,gloc[1]+1,:]=0
    fi=np.zeros((I,J,K))
    fi[gloc[0],gloc[1],:] = img[gloc[0],gloc[1],:]
    fi[floc[0],floc[1],:] = -img[gloc[0],gloc[1],:]

    def msiter(niter=100):
        nonlocal pim, v, w_E,w_S,w_SE,w_NE
        for i in range(niter):
            m_E = np.multiply(w_E.reshape(I,J,1), pim[1:-1, 2:, :]-pim[1:-1, 1:-1, :] + gamma * v[1:-1, 2:, :]-gamma * v[1:-1,1:-1,:])
            m_S = np.multiply(w_S.reshape(I,J,1), pim[2:, 1:-1, :]-pim[1:-1, 1:-1, :] + gamma * v[2:, 1:-1, :]-gamma * v[1:-1,1:-1,:])
            m_SE = np.multiply(w_SE.reshape(I,J,1),pim[2:, 2:, :] - pim[1:-1, 1:-1, :]+ gamma * v[2:, 2:, :]-gamma * v[1:-1,1:-1,:])
            m_NE = np.multiply(w_NE.reshape(I,J,1),pim[:-2, 2:, :] - pim[1:-1, 1:-1,:]+ gamma * v[:-2, 2:, :]-gamma * v[1:-1,1:-1,:])
            Io=m_E+m_S+m_SE+m_NE
            Io[:,1:,:]-=m_E[:,:-1,:]
            Io[1:,:,:]-=m_S[:-1,:,:]
            Io[1:,1:,:]-=m_SE[:-1,:-1,:]
            Io[:-1,1:,:]-=m_NE[1:,:-1,:]
            Io-=fi
            Io[gloc[0],gloc[1],:]=0
            Io[floc[0],floc[1],:]=0
            v[1:-1,1:-1,:] = gamma * v[1:-1,1:-1,:] + a * Io
            pim += v
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
    # msconti(mcont=mcont)
    return pim[1:-1,1:-1,:]
    # return img


if __name__ == "__main__":
    #E,S,W,N,SE,SW,NW,NE
    lij=((0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,-1),(-1,1))

    data_path = os.path.join(os.getcwd(), 'photos')
    im_flist = os.listdir(data_path)
    # im_no = 1
    im_no = 3
    im = mpimg.imread(os.path.join(data_path, im_flist[im_no]))
    # im=im[110:150,140:190,:]
    # im= im[400:410, 610:625, :]
    # im= im[400:440, 610:650, :]
    im=im[40:80,10:60,:]
    # im = im[40:60, 10:50, :]
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