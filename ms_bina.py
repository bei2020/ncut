import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from segment import edge_weight
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
    v = np.zeros((I+2,J+2))
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
            np.save(f, np.stack((w_E,w_S,w_SE,w_NE)))
    else:
        a = rsig ** 2 / (K + 2)
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
    floc=np.argmin(np.stack((w_E,w_S,w_SE,w_NE))[:,gloc[0],gloc[1]])
    floc=(gloc[0]+lij[floc][0],gloc[1]+lij[floc][1])
    print('floc %d %d'%(floc[0],floc[1]))
    pim= np.zeros((I+2,J+2))
    pim[floc[0]+1,floc[1]+1] = 1
    pim[gloc[0]+1,gloc[1]+1] = -1

    for w in (w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW):
        w[gloc[0],gloc[1]]=0
        w[floc[0],floc[1]]=0

    def ms_seq(niter=1):
        """Return binary img."""
        nonlocal pim,  w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW,gloc,floc,di
        gc=((gloc[0],floc[0])[gloc[0]>floc[0]],(gloc[1],floc[1])[gloc[1]>floc[1]])
        print('top left loc of gf %d %d'%(gc[0],gc[1]))
        Io = (np.multiply(w_E, pim[1:-1, 2:]) + np.multiply(w_S, pim[2:, 1:-1]) + np.multiply(w_W, pim[1:-1, :J]) \
              + np.multiply( w_N, pim[:I, 1:-1]) + np.multiply(w_SE, pim[2:, 2:]) + np.multiply(w_NE, pim[:I, 2:]) \
              + np.multiply( w_NW, pim[:I, :J]) + np.multiply(w_SW, pim[2:, :J]))/di
        gp = Io - pim[1:-1,1:-1]
        Io= np.zeros((I,J))
        Io[floc[0],floc[1]] = 1
        Io[gloc[0],gloc[1]] = -1
        im=copy.deepcopy(Io)
        Io[gc[0]:gc[0]+2,gc[1]:gc[1]+2] = np.tanh(gp/rsig)[gc[0]:gc[0]+2,gc[1]:gc[1]+2]
        Io[floc[0],floc[1]] = 1
        Io[gloc[0],gloc[1]] = -1
        ax = plt.subplot(111)
        plt.imshow(Io)
        ax.set_title('pim')
        plt.colorbar(orientation='horizontal')
        plt.show()
        for n in range(niter):
            c=[*gc]#-->
            while c[1]<J-1:
                Io[c[0],c[1]+1]=(w_W[c[0],c[1]+1]*Io[tuple(c)]+w_SW[c[0],c[1]+1]*Io[c[0]+1,c[1]])/di[c[0],c[1]+1]
                Io[c[0]+1,c[1]+1]=(w_W[c[0]+1,c[1]+1]*Io[c[0]+1,c[1]]+w_NW[c[0]+1,c[1]+1]*Io[tuple(c)]+w_N[c[0]+1,c[1]+1]*Io[c[0],c[1]+1])/di[c[0]+1,c[1]+1]
                c[1]+=1
            c=[gc[0]+1,gc[1]]
            while c[0]<I-1:
                Io[c[0]+1,c[1]]=(w_N[c[0]+1,c[1]]*Io[tuple(c)]+w_NE[c[0]+1,c[1]]*Io[c[0],c[1]+1])/di[c[0]+1,c[1]]
                while c[1]<J-1:
                    Io[c[0]+1,c[1]+1]=(w_W[c[0]+1,c[1]+1]*Io[c[0]+1,c[1]]+w_NW[c[0]+1,c[1]+1]*Io[tuple(c)]+w_N[c[0]+1,c[1]+1]*Io[c[0],c[1]+1])/di[c[0]+1,c[1]+1]
                    c[1]+=1
                c[0]+=1
                c[1]=gc[1]
            c=[*gc]
            while c[0]>0:
                Io[c[0]-1,c[1]]=(w_S[c[0]-1,c[1]]*Io[tuple(c)]+w_SE[c[0]-1,c[1]]*Io[c[0],c[1]+1])/di[c[0]-1,c[1]]
                while c[1]<J-1:
                    Io[c[0]-1,c[1]+1]=(w_W[c[0]-1,c[1]+1]*Io[c[0]-1,c[1]]+w_SW[c[0]-1,c[1]+1]*Io[tuple(c)]+w_S[c[0]-1,c[1]+1]*Io[c[0],c[1]+1])/di[c[0]-1,c[1]+1]
                    c[1]+=1
                c[0]-=1
                c[1]=gc[1]
            c=[*gc]#<--
            while c[1]>0:
                Io[c[0],c[1]-1]=(w_E[c[0],c[1]-1]*Io[tuple(c)]+w_SE[c[0],c[1]-1]*Io[c[0]+1,c[1]])/di[c[0],c[1]-1]
                Io[c[0]+1,c[1]-1]=(w_E[c[0]+1,c[1]-1]*Io[c[0]+1,c[1]]+w_NE[c[0]+1,c[1]-1]*Io[tuple(c)]+w_N[c[0]+1,c[1]-1]*Io[c[0],c[1]-1])/di[c[0]+1,c[1]-1]
                c[1]-=1
            c=[gc[0]+1,gc[1]]
            while c[0]<I-1:
                Io[c[0]+1,c[1]]=(w_N[c[0]+1,c[1]]*Io[tuple(c)]+w_NW[c[0]+1,c[1]]*Io[c[0],c[1]-1])/di[c[0]+1,c[1]]
                while c[1]>0:
                    Io[c[0]+1,c[1]-1]=(w_E[c[0]+1,c[1]-1]*Io[c[0]+1,c[1]]+w_NE[c[0]+1,c[1]-1]*Io[tuple(c)]+w_N[c[0]+1,c[1]-1]*Io[c[0],c[1]-1])/di[c[0]+1,c[1]-1]
                    c[1]-=1
                c[0]+=1
                c[1]=gc[1]
            c=[*gc]
            while c[0]>0:
                Io[c[0]-1,c[1]]=(w_S[c[0]-1,c[1]]*Io[tuple(c)]+w_SW[c[0]-1,c[1]]*Io[c[0],c[1]-1])/di[c[0]-1,c[1]]
                while c[1]>0:
                    Io[c[0]-1,c[1]-1]=(w_E[c[0]-1,c[1]-1]*Io[c[0]-1,c[1]]+w_SE[c[0]-1,c[1]-1]*Io[tuple(c)]+w_S[c[0]-1,c[1]-1]*Io[c[0],c[1]-1])/di[c[0]-1,c[1]-1]
                    c[1]-=1
                c[0]-=1
                c[1]=gc[1]

            gp = Io - im
            Io = np.tanh(gp/rsig)
            Io[floc[0],floc[1]] = 1
            Io[gloc[0],gloc[1]] = -1
            im=copy.deepcopy(Io)
            ax = plt.subplot(111)
            plt.imshow(Io)
            ax.set_title('Io')
            plt.colorbar(orientation='horizontal')
            plt.show()
        return Io

    def msiter(niter=1):
        nonlocal pim, v, Vp, w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW, cur,rsig,di
        for i in range(niter):
            # pim[1:-1,1:-1,:]-=pim[1:-1,1:-1,:][gloc[0],gloc[1],:]
            Io = (np.multiply(w_E, pim[1:-1, 2:]) + np.multiply(w_S, pim[2:, 1:-1]) + np.multiply(w_W, pim[1:-1, :J]) \
                  + np.multiply( w_N, pim[:I, 1:-1]) + np.multiply(w_SE, pim[2:, 2:]) + np.multiply(w_NE, pim[:I, 2:]) \
                  + np.multiply( w_NW, pim[:I, :J]) + np.multiply(w_SW, pim[2:, :J]))/di.reshape((I, J))
            gp = Io - pim[1:-1,1:-1]
            # v[1:-1,1:-1] = gamma * v[1:-1,1:-1] + a *  gp
            pim[1:-1,1:-1] = np.tanh(gp/rsig)
            pim[floc[0]+1,floc[1]+1] = 1
            pim[gloc[0]+1,gloc[1]+1] = -1
        ax = plt.subplot(111)
        plt.imshow(pim[1:-1,1:-1])
        ax.set_title('pim')
        plt.colorbar(orientation='horizontal')
        plt.show()

    def msconti(ncont=0, mcont=12):
        nonlocal pim
        imgc = copy.deepcopy(pim)
        msiter(10)
        if np.array_equal(np.round(np.exp(pim),8),np.round(np.exp(imgc),8)):
        # ginf=1e-12
        # if np.sum(np.square((pim-imgc)[1:-1,1:-1,:]))<ginf:
            print('fix found continue iter %d' % (ncont * 1000))
            # print(np.round(pim[:10, :10, 0], 6))
            # print(np.round(v[1:-1, 1:-1, 0], 6))
            # print('ground ',pim[gloc[0],gloc[1],:])
            return
        else:
            if ncont == mcont:
                print('no fix after continue iter %d' % (ncont * 1000))
                # print(np.round(pim[:10, :10, 0], 6))
                # print(np.round(imgc[:10, :10, 0], 6))
                # print('ground ',img[gloc[0],gloc[1],:])
                # print(np.round(np.exp(img[:10,:10,0]), 6))
                # print(np.round(np.exp(imgc[:10,:10,0]), 6))
                return
            print('continue')
            ncont += 1
            msiter(niter=1)
            msconti(ncont, mcont)

    # msiter()
    # msconti(mcont=mcont)
    img=ms_seq()
    # return pim[1:-1,1:-1]
    # return pim[1:-1,1:-1,:]
    return img


def grad_m(grad):
    """Return prob mean value of gradient"""
    h, b = np.histogram(abs(grad).flatten(), bins=20)
    pk = np.argmax(h)
    nf = 3
    ginf = 1e-8
    vh = [None] * 2
    vh[0] = np.dot(h[pk:pk + nf], b[1 + pk:1 + pk + nf]) / np.sum(h[pk:pk + nf])
    vh[1] = np.dot(h[pk:pk + nf + 1], b[1 + pk:2 + pk + nf]) / np.sum(h[pk:pk + nf + 1])
    for i in range(2, len(h)):
        v = np.dot(h[pk:pk + nf + i], b[1 + pk:1 + pk + nf + i]) / np.sum(h[pk:pk + nf + i])
        if (abs(v - vh[1]) < ginf) & (abs(vh[0] + v - 2 * vh[1]) < ginf):
            break
        vh[0] = vh[1]
        vh[1] = v
    return np.round(v, int(np.ceil(np.log10(1 / v))))


if __name__ == "__main__":
    lij=((0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,-1),(-1,1))

    with open('imgs/tri_part','rb') as f:
        im=np.load(f)
    im=im[4:14,5:15]
    # iph=im.reshape((*im.shape,1))
    iph=(im/np.sum(im)).reshape((*im.shape,1))
    # iph=-np.log(im/np.sum(im)).reshape((*im.shape,1))
    # iph=-np.log(iph).reshape((*im.shape,1))

    # data_path = os.path.join(os.getcwd(), 'photos')
    # im_flist = os.listdir(data_path)
    # im_no = 0
    # im = mpimg.imread(os.path.join(data_path, im_flist[im_no]))
    # # im=im[110:150,140:190,:]
    # im = im[40:60, 10:50, :]
    # # im=im[40:80,10:60,:]
    # ime = np.einsum('ijk->k', im.astype('uint32')).reshape(1, 1, im.shape[2])
    # iph = im / ime
    # iph[iph == 0] = .0000001 / np.sum(ime)
    # # iph=np.sqrt(iph)

    I, J, K = iph.shape
    l = np.arange(I * J).reshape(I, J)
    # ig = msimg(iph, mcont=4)
    ig=msimg(copy.deepcopy(iph),mcont=1)
    # ig=msimg(ig,rsig=.08)

    # blabels=np.zeros((I,J,6))
    # prob = 1 / (1 + np.exp(-np.sum(np.multiply(ig,iph),axis=2)))
    # blabels[:,:,0] = (prob > np.random.rand(*prob.shape)).astype('int')

    ax = plt.subplot(231)
    # plt.imshow(im[:,:,0])
    plt.imshow(im)
    ax.set_title('sample')
    plt.colorbar(orientation='horizontal')
    ax = plt.subplot(232)
    # plt.imshow(np.square(ig[:,:,0])*np.sum(im))
    # plt.imshow(ig[:, :, 0])
    plt.imshow(ig)
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

    # cmsk=np.sum(blabels,-1).astype('bool') # cut nodes mask
    # iph[cmsk] = -np.inf
    # iph[cmsk] = 0
    # ig=msimg(copy.deepcopy(iph),rsig=.1,mcont=4)
    # ig[cmsk] = -np.inf
    # prob = 1 / (1 + np.exp(-ig[:,:,ch]))
    # blabels[:,:,1] = (prob > np.random.rand(*prob.shape)).astype('int')
    #
    # ax=plt.subplot(234)
    # # plt.imshow(ig[:,:,0])
    # plt.imshow(ig[:,:,0]/np.sum(ig[:,:,0])*np.sum(im))
    # # ax.set_title('ch%d iph01'%ch)
    # plt.colorbar(orientation='horizontal')
    # ax=plt.subplot(235)
    # plt.imshow(blabels[:,:,1])
    # ax.set_title('seg01 .1 4000')
    # plt.colorbar(orientation='horizontal')
    #
    # cmsk=np.sum(blabels,-1).astype('bool')
    # iph[cmsk] = -np.inf
    # iph[cmsk] = 0
    # ig=msimg(copy.deepcopy(iph),rsig=.3,mcont=2)
    # ig[cmsk] = -np.inf
    # mct=np.einsum('ijk->k',ig==np.max(ig,-1).reshape(I,J,1).astype('int'))
    # ch=np.arange(K)[mct==np.max(mct)][0]
    # prob = 1 / (1 + np.exp(-ig[:,:,ch]))
    # blabels[:,:,2] = (prob > np.random.rand(*prob.shape)).astype('int')
    #
    # ax=plt.subplot(236)
    # plt.imshow(ig[:,:,ch])
    # ax.set_title('ch%d iph001 .3 2000'%ch)
    # plt.colorbar(orientation='horizontal')

    plt.show()
