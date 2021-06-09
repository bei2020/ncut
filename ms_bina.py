import numpy as np
from matplotlib import pyplot as plt
from anisodiff2D import anisodiff2D
import os
import matplotlib.image as mpimg
from PIL import Image
from segment import edge_weight,grad_m
import copy
from time import gmtime, strftime
import logging
FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('ms')
logger.setLevel('INFO')
# logger.setLevel('DEBUG')

def SE_pe(gc,ps,Ip,w_W,w_N,w_NW,di,rsig):
    """Return heat expanded patch.
    gc: ground center,heat source in a patch.
    ps: int,patch size=ps*ps
    """
    logger.debug('pe Ip%d %d'%Ip.shape)
    for i in range(1,ps):
        Ip[0,i]+=w_W[0,i]*Ip[0,i-1]/di[0,i]
        Ip[i,0]+=w_N[i,0]*Ip[i-1,0]/di[i,0]
    #skew patch
    pa=np.zeros((2*ps-1,2*ps-1))
    for i in range(ps):
        pa[i,i:i+ps]=Ip[i,:]
    wp_W=np.zeros((2*ps-1,2*ps-1))
    for i in range(ps):
        wp_W[i,i:i+ps]=w_W[i,:]
    wp_N=np.zeros((2*ps-1,2*ps-1))
    for i in range(ps):
        wp_N[i,i:i+ps]=w_N[i,:]
    wp_NW=np.zeros((2*ps-1,2*ps-1))
    for i in range(ps):
        wp_NW[i,i:i+ps]=w_NW[i,:]
    dip=np.zeros((2*ps-1,2*ps-1))
    for i in range(ps):
        dip[i,i:i+ps]=di[i,:]

    for i in range(1,ps):
        pa[1:i+1,i+1]+=(pa[1:i+1,i]*wp_W[1:i+1,i+1]+pa[:i,i]*wp_N[1:i+1,i+1]+pa[:i,i-1]*wp_NW[1:i+1,i+1])/dip[1:i+1,i+1]
    for i in  range(1,ps):
        pa[i:,i+ps-1]+=(pa[i:,i+ps-2]*wp_W[i:,i+ps-1]+pa[i-1:-1,i+ps-2]*wp_N[i:,i+ps-1]+pa[i-1:-1,i+ps-3]*wp_NW[i:,i+ps-1])/dip[i:,i+ps-1]

    for i in range(ps):
        Ip[i,:]=pa[i,i:i+ps]
    Ip = np.tanh(Ip/rsig)
    return Ip

def SE_ps(ground_center,Im,ps,pr,w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW,di,rsig,I,J):
    """Return SE direction heat expanded image.
    ps: int, sequence fixed length*2,patch size=ps*ps."""
    c=[*ground_center]
    nsi=(I-c[0])//pr
    nsj=(J-c[1])//pr
    logger.info('pgc %d %d,nsi %d,nsj %d,asig %f'%(*ground_center,nsi,nsj,rsig))
    for i in range(1,nsi):
        for j in range(1,nsj):
            logger.debug('ploc %d %d'%(c[0],c[1]))
            Im[c[0]:c[0]+ps,c[1]:c[1]+ps]=SE_pe(c,ps,Im[c[0]:c[0]+ps,c[1]:c[1]+ps],w_W[c[0]:c[0]+ps,c[1]:c[1]+ps],w_N[c[0]:c[0]+ps,c[1]:c[1]+ps],w_NW[c[0]:c[0]+ps,c[1]:c[1]+ps],di[c[0]:c[0]+ps,c[1]:c[1]+ps],rsig)
            c[1]+=pr
        if c[1]<J-pr:
            logger.debug('p margin loc %d %d'%(c[0],c[1]))
            psm=J-c[1]
            Im[c[0]:c[0]+psm,c[1]:c[1]+psm]=SE_pe(c,J-c[1],Im[c[0]:c[0]+psm,c[1]:c[1]+psm],w_W[c[0]:c[0]+psm,c[1]:c[1]+psm],w_N[c[0]:c[0]+psm,c[1]:c[1]+psm],w_NW[c[0]:c[0]+psm,c[1]:c[1]+psm],di[c[0]:c[0]+psm,c[1]:c[1]+psm],rsig)
        c[0]+=pr
        c[1]=ground_center[1]
        ax = plt.subplot(111)
        plt.imshow(Im)
        ax.set_title('Iop')
        plt.colorbar(orientation='horizontal')
        plt.show()
    if c[0]<I-pr:
        logger.debug('p margin loc %d %d'%(c[0],c[1]))
        psm=I-c[0]
        prm=psm//2
        nsj=(J-c[1])//prm
        for j in range(1,nsj):
            Im[c[0]:c[0]+psm,c[1]:c[1]+psm]=SE_pe(c,psm,Im[c[0]:c[0]+psm,c[1]:c[1]+psm],w_W[c[0]:c[0]+psm,c[1]:c[1]+psm],w_N[c[0]:c[0]+psm,c[1]:c[1]+psm],w_NW[c[0]:c[0]+psm,c[1]:c[1]+psm],di[c[0]:c[0]+psm,c[1]:c[1]+psm],rsig)
            c[1]+=prm
        if c[1]<J-prm:
            logger.debug('p margin loc %d %d'%(c[0],c[1]))
            psm=J-c[1]
            Im[c[0]:c[0]+psm,c[1]:c[1]+psm]=SE_pe(c,J-c[1],Im[c[0]:c[0]+psm,c[1]:c[1]+psm],w_W[c[0]:c[0]+psm,c[1]:c[1]+psm],w_N[c[0]:c[0]+psm,c[1]:c[1]+psm],w_NW[c[0]:c[0]+psm,c[1]:c[1]+psm],di[c[0]:c[0]+psm,c[1]:c[1]+psm],rsig)
        # ax = plt.subplot(111)
        # plt.imshow(Im)
        # ax.set_title('Iopm')
        # plt.colorbar(orientation='horizontal')
        # plt.show()
    return Im

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
    floc=np.argmin(np.stack((w_E,w_S,w_W,w_N,w_SE,w_SW,w_NW,w_NE))[:,gloc[0],gloc[1]])
    floc=(gloc[0]+lij[floc][0],gloc[1]+lij[floc][1])
    print('floc %d %d'%(floc[0],floc[1]))
    asig=np.sum(abs(img[gloc]-img[floc]),-1)/K

    for w in (w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW):
        w[gloc[0],gloc[1]]=0
        w[floc[0],floc[1]]=0

    def ms_seq():
        """Return binary img."""
        nonlocal w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW,gloc,floc,di,asig
        # umax=.5
        ps=6
        pr=ps//2
        Io= np.zeros((I,J))
        Io[floc[0],floc[1]] = 1
        Io[gloc[0],gloc[1]] = -1
        # g f heat expand
        Io=SE_ps(gloc,Io,ps,pr,w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW,di,asig,I,J)
        Io=SE_ps(floc,Io,ps,pr,w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW,di,asig,I,J)

        Io = np.tanh(Io/asig)
        ax = plt.subplot(111)
        plt.imshow(Io)
        ax.set_title('Io')
        plt.colorbar(orientation='horizontal')
        plt.show()
        return Io

    img=ms_seq()
    return img


if __name__ == "__main__":
    #E,S,W,N,SE,SW,NW,NE
    lij=((0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,-1),(-1,1))

    # with open('imgs/tri_part','rb') as f:
    #     im=np.load(f)
    # # im=im[4:14,5:15]
    # # im=im[5:,5:15]
    # # iph=im.reshape((*im.shape,1))
    # iph=(im/np.sum(im)).reshape((*im.shape,1))

    data_path = os.path.join(os.getcwd(), 'photos')
    im_flist = os.listdir(data_path)
    im_no = 0
    im = mpimg.imread(os.path.join(data_path, im_flist[im_no]))
    im = im[40:60, 10:50, :]
    # im = Image.open(os.path.join(data_path, im_flist[im_no]))
    # im = im.resize((np.array(im.size) / 10).astype(int))
    # im = np.asarray(im)
    # im = im[10:22, 10:50,:]
    ime = np.einsum('ijk->k', im.astype('uint32')).reshape(1, 1, im.shape[2])
    iph = im / ime

    # I, J, K = iph.shape
    # l = np.arange(I * J).reshape(I, J)
    # # ig = msimg(iph, mcont=4)
    ig=msimg(copy.deepcopy(iph))
    # # ig=msimg(ig,rsig=.08)

    ax = plt.subplot(121)
    # plt.imshow(im[:,:,0])
    plt.imshow(im)
    ax.set_title('sample')
    plt.colorbar(orientation='horizontal')
    ax = plt.subplot(122)
    # plt.imshow(np.square(ig[:,:,0])*np.sum(im))
    plt.imshow(ig)
    # plt.imshow(blabels,cmap='gray')
    ax.set_title('one part')
    plt.colorbar(orientation='horizontal')

    # ax = plt.subplot(233)
    # plt.imshow(blabels[:, :, 0])
    # ax.set_title('ib')
    # plt.colorbar(orientation='horizontal')

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
