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
    if init_wt == 1:
        # 1234:ESWN, Io: DHWC, D=8 directions
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
        # fn = 'wt_rsig%f_%s' % (rsig, fn)
        # with open('%s.npy' % fn, 'wb') as f:
        #     np.save(f, np.stack((w_E,w_S,w_SE,w_NE)))

    else:
        fn = 'wt_rsig%f_%s' % (rsig, fn)
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
    floc=np.argmin(np.stack((w_E,w_S,w_W,w_N,w_SE,w_SW,w_NW,w_NE))[:,gloc[0],gloc[1]])
    floc=(gloc[0]+lij[floc][0],gloc[1]+lij[floc][1])
    print('floc %d %d'%(floc[0],floc[1]))

    for w in (w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW):
        w[gloc[0],gloc[1]]=0
        w[floc[0],floc[1]]=0

    pim = np.ones((I+2,J+2)).astype('int')
    pim[gloc[0]+1,gloc[1]+1] = 0
    pim[1:-1,1:-1][np.sum(abs(img-img[gloc]),-1)<np.sum(abs(img-img[floc]),-1)]=0 #gloc=0
    def msiter(niter=1000):
        nonlocal pim, w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW
        for i in range(niter):
            e1 = w_E*(pim[1:-1, 2:]^1)+ w_S*(pim[2:, 1:-1]^1)+w_W*(pim[1:-1, :J]^1)+w_N*(pim[:I, 1:-1]^1)\
                 +w_SE*(pim[2:, 2:]^1)+ w_NE*(pim[:I, 2:]^1)+w_NW*(pim[:I, :J]^1) + w_SW*(pim[2:, :J]^1)/di
            e0 = w_E*(pim[1:-1, 2:]^0)+ w_S*(pim[2:, 1:-1]^0)+w_W*(pim[1:-1, :J]^0)+w_N*(pim[:I, 1:-1]^0) \
                 +w_SE*(pim[2:, 2:]^0)+ w_NE*(pim[:I, 2:]^0)+w_NW*(pim[:I, :J]^0) + w_SW*(pim[2:, :J]^0)/di
            hase=(e1+e0)!=0
            pim[1:-1,1:-1][hase]=(1/(1+np.exp((e1-e0)[hase]/(e1+e0)[hase]))>np.random.rand(I,J)[hase]).astype('int')
            # pim[1:-1,1:-1]=(1/(1+np.exp((e1-e0)))>np.random.rand(I,J)).astype('int')
            pim[gloc[0]+1,gloc[1]+1] = 0
            pim[floc[0]+1,floc[1]+1] = 1
            # print(e1[e1!=0])
            # print(e1.shape)
        return

    msiter()
    return pim[1:-1,1:-1]
    # return img


if __name__ == "__main__":
    #E,S,W,N,SE,SW,NW,NE
    lij=((0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,-1),(-1,1))

    data_path = os.path.join(os.getcwd(), 'photos')
    im_flist = os.listdir(data_path)
    # im_no = 5
    im_no = 3
    im = mpimg.imread(os.path.join(data_path, im_flist[im_no]))
    # im=im[110:150,140:190,:]
    # im= im[400:410, 610:625, :]
    # im= im[400:440, 610:650, :]
    # im = im[40:60, 10:50, :]
    # im=im[40:80,10:60,:]
    # im=im[45:55,50:60,:]
    im = im[40:60, 10:50, :]
    # im= im[360:400, 450:650, :]

    ig = msimg(im/1, mcont=0)

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
