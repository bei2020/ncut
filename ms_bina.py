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

def SE_expa(ground_center,Im,w_N,w_NE,w_W,w_NW,di):
    """gc heat expand in S,E directions"""
    c=[*ground_center]
    I,J=Im.shape
    while (c[0]<I-1) &(c[1]<J-1):
        # print('SE %d %d'%(c[0],c[1]))
        Im[c[0]+1,c[1]]+= (w_N[c[0]+1,c[1]]*Im[tuple(c)]+w_NE[c[0]+1,c[1]]*Im[c[0],c[1]+1])/di[c[0]+1,c[1]]
        while c[1]<J-1:
            Im[c[0]+1,c[1]+1]+= (w_W[c[0]+1,c[1]+1]*Im[c[0]+1,c[1]]+w_NW[c[0]+1,c[1]+1]*Im[tuple(c)]+w_N[c[0]+1,c[1]+1]*Im[c[0],c[1]+1])/di[c[0]+1,c[1]+1]
            c[1]+=1
        c[0]+=1
        c[1]=ground_center[1]

def NE_expa(ground_center,Im,w_S,w_SE,w_W,w_SW,di):
    """gc heat expand in N,E directions"""
    c=[*ground_center]
    I,J=Im.shape
    while (c[0]>0) &(c[1]<J-1):
        # print('NE %d %d'%(c[0],c[1]))
        Im[c[0]-1,c[1]]+=(w_S[c[0]-1,c[1]]*Im[tuple(c)]+w_SE[c[0]-1,c[1]]*Im[c[0],c[1]+1])/di[c[0]-1,c[1]]
        while c[1]<J-1:
            Im[c[0]-1,c[1]+1]+=(w_W[c[0]-1,c[1]+1]*Im[c[0]-1,c[1]]+w_SW[c[0]-1,c[1]+1]*Im[tuple(c)]+w_S[c[0]-1,c[1]+1]*Im[c[0],c[1]+1])/di[c[0]-1,c[1]+1]
            c[1]+=1
        c[0]-=1
        c[1]=ground_center[1]

def SW_expa(ground_center,Im,w_N,w_NW,w_E,w_NE,di):
    """gc heat expand in S,W directions"""
    c=[*ground_center]
    I,J=Im.shape
    while (c[0]<I-1) &(c[1]>0):
        # print('SW %d %d'%(c[0],c[1]))
        Im[c[0]+1,c[1]]+=(w_N[c[0]+1,c[1]]*Im[tuple(c)]+w_NW[c[0]+1,c[1]]*Im[c[0],c[1]-1])/di[c[0]+1,c[1]]
        while c[1]>0:
            Im[c[0]+1,c[1]-1]+=(w_E[c[0]+1,c[1]-1]*Im[c[0]+1,c[1]]+w_NE[c[0]+1,c[1]-1]*Im[tuple(c)]+w_N[c[0]+1,c[1]-1]*Im[c[0],c[1]-1])/di[c[0]+1,c[1]-1]
            c[1]-=1
        c[0]+=1
        c[1]=ground_center[1]

def NW_expa(ground_center,Im,w_S,w_SW,w_E,w_SE,di):
    """gc heat expand in N,W directions"""
    c=[*ground_center]
    while (c[0]>0) &(c[1]>0):
        # print('NW %d %d'%(c[0],c[1]))
        Im[c[0]-1,c[1]]+=(w_S[c[0]-1,c[1]]*Im[tuple(c)]+w_SW[c[0]-1,c[1]]*Im[c[0],c[1]-1])/di[c[0]-1,c[1]]
        while c[1]>0:
            Im[c[0]-1,c[1]-1]+=(w_E[c[0]-1,c[1]-1]*Im[c[0]-1,c[1]]+w_SE[c[0]-1,c[1]-1]*Im[tuple(c)]+w_S[c[0]-1,c[1]-1]*Im[c[0],c[1]-1])/di[c[0]-1,c[1]-1]
            c[1]-=1
        c[0]-=1
        c[1]=ground_center[1]

def heat_expa(ground_center,Im,w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW,di,rsig):
    """gf square heat expand, top left corner as ground center"""
    logger.debug('heat center %d %d'%ground_center)
    dmin=1e-2
    c=[*ground_center]#-->
    while c[1]<J-1:
        Im[c[0],c[1]+1]+=(w_W[c[0],c[1]+1]*Im[tuple(c)]+w_SW[c[0],c[1]+1]*Im[c[0]+1,c[1]])/di[c[0],c[1]+1]
        Im[c[0]+1,c[1]+1]+=(w_W[c[0]+1,c[1]+1]*Im[c[0]+1,c[1]]+w_NW[c[0]+1,c[1]+1]*Im[tuple(c)]+w_N[c[0]+1,c[1]+1]*Im[c[0],c[1]+1])/di[c[0]+1,c[1]+1]
        c[1]+=1
    SE_expa((ground_center[0]+1,ground_center[1]),Im,w_N,w_NE,w_W,w_NW,di)
    NE_expa(ground_center,Im,w_S,w_SE,w_W,w_SW,di)
    c=[*ground_center]#<--
    while c[1]>0:
        Im[c[0],c[1]-1]+=(w_E[c[0],c[1]-1]*Im[tuple(c)]+w_SE[c[0],c[1]-1]*Im[c[0]+1,c[1]])/di[c[0],c[1]-1]
        Im[c[0]+1,c[1]-1]+=(w_E[c[0]+1,c[1]-1]*Im[c[0]+1,c[1]]+w_NE[c[0]+1,c[1]-1]*Im[tuple(c)]+w_N[c[0]+1,c[1]-1]*Im[c[0],c[1]-1])/di[c[0]+1,c[1]-1]
        c[1]-=1
    SW_expa((ground_center[0]+1,ground_center[1]),Im,w_N,w_NW,w_E,w_NE,di)
    NW_expa(ground_center,Im,w_S,w_SW,w_E,w_SE,di)

    Im[Im>1]=1
    Im[Im<-1]=-1
    Im[(1-Im)<dmin]=1
    Im[(Im+1)<dmin]=-1

    # Im = np.tanh(Im/rsig)
    ax = plt.subplot(111)
    plt.imshow(Im)
    ax.set_title('Im')
    plt.colorbar(orientation='horizontal')
    plt.show()
    return Im

def split_expa(ground_center,Im,w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW,di,rsig):
    """Return new ground_center.
    Find corners with the same heat of g f as new gc, calculate new gf square , then heat expand."""
    logger.debug('split %d %d'%ground_center)
    c=[*ground_center]
    c[0]-=np.sum((Im[:c[0],c[1]]==Im[tuple(c)]).astype('int'))
    c[1]-=np.sum((Im[c[0],:c[1]]==Im[tuple(c)]).astype('int'))
    gc1=tuple(c)
    logger.debug('top left gc1 %d %d'%(c[0],c[1]))
    if (gc1[0]>0) & (gc1[1]>0):
        if (Im[c[0]-1,c[1]]!=-Im[tuple(c)])&(Im[c[0],c[1]-1]!=-Im[tuple(c)]):
            Im[c[0]+1,c[1]]+= (w_N[c[0]+1,c[1]]*Im[tuple(c)]+w_NE[c[0]+1,c[1]]*Im[c[0],c[1]+1])/di[c[0]+1,c[1]]
            Im[c[0]+1,c[1]+1]+= (w_W[c[0]+1,c[1]+1]*Im[c[0]+1,c[1]]+w_NW[c[0]+1,c[1]+1]*Im[tuple(c)]+w_N[c[0]+1,c[1]+1]*Im[c[0],c[1]+1])/di[c[0]+1,c[1]+1]
            Im[c[0]+1,c[1]] = np.tanh(Im[c[0]+1,c[1]]/rsig)
            Im[c[0]+1,c[1]+1] = np.tanh(Im[c[0]+1,c[1]+1]/rsig)
            Im=heat_expa(gc1,Im,w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW,di,rsig)

    c=[ground_center[0],ground_center[1]+1]
    c[0]-=np.sum((Im[:c[0],c[1]]==Im[tuple(c)]).astype('int'))
    c[1]+=np.sum((Im[c[0],c[1]+1:]==Im[tuple(c)]).astype('int'))
    gc2=tuple(c)
    logger.debug('top right gc2 %d %d'%(c[0],c[1]))
    if (gc2[0]>0) & (gc2[1]<J-1):
        if (Im[c[0]-1,c[1]]!=-Im[tuple(c)])&(Im[c[0],c[1]+1]!=-Im[tuple(c)]):
            Im[c[0]+1,c[1]]+=(w_N[c[0]+1,c[1]]*Im[tuple(c)]+w_NW[c[0]+1,c[1]]*Im[c[0],c[1]-1])/di[c[0]+1,c[1]]
            Im[c[0]+1,c[1]-1]+=(w_E[c[0]+1,c[1]-1]*Im[c[0]+1,c[1]]+w_NE[c[0]+1,c[1]-1]*Im[tuple(c)]+w_N[c[0]+1,c[1]-1]*Im[c[0],c[1]-1])/di[c[0]+1,c[1]-1]
            Im[c[0]+1,c[1]] = np.tanh(Im[c[0]+1,c[1]]/rsig)
            Im[c[0]+1,c[1]-1] = np.tanh(Im[c[0]+1,c[1]-1]/rsig)
            Im=heat_expa((gc2[0],gc2[1]-1),Im,w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW,di,rsig)

    c=[ground_center[0]+1,ground_center[1]]
    c[0]+=np.sum((Im[c[0]+1:,c[1]]==Im[tuple(c)]).astype('int'))
    c[1]-=np.sum((Im[c[0],:c[1]]==Im[tuple(c)]).astype('int'))
    gc3=tuple(c)
    logger.debug('bottom left gc3 %d %d'%(c[0],c[1]))
    if (gc3[0]<I-1) & (gc3[1]>0):
        if (Im[c[0]+1,c[1]]!=-Im[tuple(c)])&(Im[c[0],c[1]-1]!=-Im[tuple(c)]):
            Im[c[0]-1,c[1]]+=(w_S[c[0]-1,c[1]]*Im[tuple(c)]+w_SE[c[0]-1,c[1]]*Im[c[0],c[1]+1])/di[c[0]-1,c[1]]
            Im[c[0]-1,c[1]+1]+=(w_W[c[0]-1,c[1]+1]*Im[c[0]-1,c[1]]+w_SW[c[0]-1,c[1]+1]*Im[tuple(c)]+w_S[c[0]-1,c[1]+1]*Im[c[0],c[1]+1])/di[c[0]-1,c[1]+1]
            Im[c[0]-1,c[1]] = np.tanh(Im[c[0]-1,c[1]]/rsig)
            Im[c[0]-1,c[1]+1] = np.tanh(Im[c[0]-1,c[1]+1]/rsig)
            Im=heat_expa((gc3[0]-1,gc3[1]),Im,w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW,di,rsig)

    c=[ground_center[0]+1,ground_center[1]+1]
    c[0]+=np.sum((Im[c[0]+1:,c[1]]==Im[tuple(c)]).astype('int'))
    c[1]+=np.sum((Im[c[0],c[1]+1:]==Im[tuple(c)]).astype('int'))
    gc4=tuple(c)
    logger.debug('bottom right gc4 %d %d'%(c[0],c[1]))
    if (gc4[0]<I-1) & (gc4[1]<J-1):
        if (Im[c[0]+1,c[1]]!=-Im[tuple(c)])&(Im[c[0],c[1]+1]!=-Im[tuple(c)]):
            Im[c[0]-1,c[1]]+=(w_S[c[0]-1,c[1]]*Im[tuple(c)]+w_SW[c[0]-1,c[1]]*Im[c[0],c[1]-1])/di[c[0]-1,c[1]]
            Im[c[0]-1,c[1]-1]+=(w_E[c[0]-1,c[1]-1]*Im[c[0]-1,c[1]]+w_SE[c[0]-1,c[1]-1]*Im[tuple(c)]+w_S[c[0]-1,c[1]-1]*Im[c[0],c[1]-1])/di[c[0]-1,c[1]-1]
            Im[c[0]-1,c[1]] = np.tanh(Im[c[0]-1,c[1]]/rsig)
            Im[c[0]-1,c[1]-1] = np.tanh(Im[c[0]-1,c[1]-1]/rsig)
            Im=heat_expa((gc4[0]-1,gc4[1]-1),Im,w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW,di,rsig)

    Im = np.tanh(Im/rsig)
    Im[np.invert(((Im == 1).astype('int') + (Im == -1).astype('int')).astype('bool'))]=0
    return Im,(gc1,gc2,gc3,gc4)


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

    for w in (w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW):
        w[gloc[0],gloc[1]]=0
        w[floc[0],floc[1]]=0

    def ms_seq():
        """Return binary img."""
        nonlocal w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW,gloc,floc,di,rsig
        # umax=.5
        pim= np.zeros((I+2,J+2))
        pim[floc[0]+1,floc[1]+1] = 1
        pim[gloc[0]+1,gloc[1]+1] = -1
        gc=((gloc[0],floc[0])[gloc[0]>floc[0]],(gloc[1],floc[1])[gloc[1]>floc[1]])
        print('top left loc of gf %d %d'%(gc[0],gc[1]))
        Io = (np.multiply(w_E, pim[1:-1, 2:]) + np.multiply(w_S, pim[2:, 1:-1]) + np.multiply(w_W, pim[1:-1, :J]) \
              + np.multiply( w_N, pim[:I, 1:-1]) + np.multiply(w_SE, pim[2:, 2:]) + np.multiply(w_NE, pim[:I, 2:]) \
              + np.multiply( w_NW, pim[:I, :J]) + np.multiply(w_SW, pim[2:, :J]))/di
        gp = Io + pim[1:-1,1:-1]
        Io= np.zeros((I,J))
        Io[floc[0],floc[1]] = 1
        Io[gloc[0],gloc[1]] = -1
        # im=copy.deepcopy(Io)
        Io[gc[0]:gc[0]+2,gc[1]:gc[1]+2] = np.tanh(gp/rsig)[gc[0]:gc[0]+2,gc[1]:gc[1]+2]
        Io[floc[0],floc[1]] = 1
        Io[gloc[0],gloc[1]] = -1
        ax = plt.subplot(111)
        plt.imshow(Io)
        ax.set_title('pim')
        plt.colorbar(orientation='horizontal')
        plt.show()

        # g f heat expand
        Io=heat_expa(gc,Io,w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW,di,rsig)

        Io = np.tanh(Io/rsig)
        ax = plt.subplot(111)
        plt.imshow(Io)
        ax.set_title('Io')
        plt.colorbar(orientation='horizontal')
        plt.show()

        # new heat center
        Io,gcc=split_expa(gc,Io,w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW,di,rsig)
        for gc in gcc:
            if (gc[0]>0) &(gc[0]<I-1) & (gc[1]>0) &(gc[1]<J-1):
                Io,gcs=split_expa(gc,Io,w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW,di,rsig)
                # for g in gcs:
                #     if (g[0]>0) &(g[0]<I-1) & (g[1]>0) &(g[1]<J-1):
                #         Io,gs=split_expa(g,Io,w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW,di,rsig)
                #         for g2 in gs:
                #             if (g2[0]>0) &(g2[0]<I-1) & (g2[1]>0) &(g2[1]<J-1):
                #                 g2s=split_expa(g,Io,w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW,di)
            #         else:
            #             print('stop g %d %d'%g)
            # else:
            #     print('stop gc %d %d'%gc)

        # Io[Io>1]=1
        # Io[Io<-1]=-1
        # Io[(1-Io)<dmin]=1
        # Io[(Io+1)<dmin]=-1

        return Io

    def edge_seq():
        nonlocal w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW,gloc,floc,di,rsig
        def n_expa(c,Im,w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW,di,rsig,I,J):
            """c: center, left pixel of an edge, tuple"""
            if (abs(Im[c[0]-1,c[1]])==1) & (abs(Im[c[0]-1,c[1]+1])==1):
                return

            Im[c[0]-1,c[1]]+=(w_S[c[0]-1,c[1]]*Im[c]+w_SE[c[0]-1,c[1]]*Im[c[0],c[1]+1])/di[c[0]-1,c[1]]
            Im[c[0]-1,c[1]+1]+=(w_SW[c[0]-1,c[1]+1]*Im[c]+w_S[c[0]-1,c[1]+1]*Im[c[0],c[1]+1]+w_W[c[0]-1,c[1]+1]*Im[c[0]-1,c[1]])/di[c[0]-1,c[1]+1]
            Im[c[0]-1,c[1]] = np.tanh(Im[c[0]-1,c[1]]/rsig)
            Im[c[0]-1,c[1]+1] = np.tanh(Im[c[0]-1,c[1]+1]/rsig)
            logger.debug('n %d %d'%c)
            if np.logical_xor(Im[c[0]-1,c[1]]>0,Im[c[0]-1,c[1]+1]>0) &(c[0]>1):
                n_expa((c[0]-1,c[1]),Im,w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW,di,rsig,I,J)
            if  np.logical_xor(Im[c[0]-1,c[1]]>0,Im[c]>0) &(c[1]>0):
                w_expa((c[0]-1,c[1]),Im,w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW,di,rsig,I,J)
            if  np.logical_xor(Im[c[0],c[1]+1]>0,Im[c[0]-1,c[1]+1]>0):
                e_expa((c[0]-1,c[1]+1),Im,w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW,di,rsig,I,J)
            return

        def s_expa(c,Im,w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW,di,rsig,I,J):
            """c: center, left pixel of an edge, tuple"""
            if (abs(Im[c[0]+1,c[1]])==1) & (abs(Im[c[0]+1,c[1]+1])==1):
                return
            Im[c[0]+1,c[1]]+=(w_N[c[0]+1,c[1]]*Im[c]+w_NE[c[0]+1,c[1]]*Im[c[0],c[1]+1])/di[c[0]+1,c[1]]
            Im[c[0]+1,c[1]+1]+=(w_NW[c[0]+1,c[1]+1]*Im[c]+w_N[c[0]+1,c[1]+1]*Im[c[0],c[1]+1]+w_W[c[0]+1,c[1]+1]*Im[c[0]+1,c[1]])/di[c[0]+1,c[1]+1]
            Im[c[0]+1,c[1]] = np.tanh(  Im[c[0]+1,c[1]]/rsig)
            Im[c[0]+1,c[1]+1] = np.tanh(Im[c[0]+1,c[1]+1]/rsig)
            logger.debug('s %d %d'%c)
            if np.logical_xor(Im[c[0]+1,c[1]]>0,Im[c[0]+1,c[1]+1]>0) &(c[0]<I-2):
                s_expa((c[0]+1,c[1]),Im,w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW,di,rsig,I,J)
            if np.logical_xor(Im[c]>0,Im[c[0]+1,c[1]]>0) &(c[1]>0):
                w_expa(c,Im,w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW,di,rsig,I,J)
            if np.logical_xor(Im[c[0],c[1]+1]>0,Im[c[0]+1,c[1]+1]>0) &(c[1]<J-1):
                e_expa((c[0],c[1]+1),Im,w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW,di,rsig,I,J)
            return

        def w_expa(c,Im,w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW,di,rsig,I,J):
            """c: center, top pixel of an edge, tuple"""
            if (abs(Im[c[0],c[1]-1])==1) & (abs(Im[c[0]+1,c[1]-1])==1):
                return
            Im[c[0],c[1]-1]+=(w_E[c[0],c[1]-1]*Im[c]+w_SE[c[0],c[1]-1]*Im[c[0]+1,c[1]])/di[c[0],c[1]-1]
            Im[c[0]+1,c[1]-1]+=(w_NE[c[0]+1,c[1]-1]*Im[c]+w_E[c[0]+1,c[1]-1]*Im[c[0]+1,c[1]]+w_N[c[0]+1,c[1]-1]*Im[c[0],c[1]-1])/di[c[0]+1,c[1]-1]
            Im[c[0],c[1]-1] = np.tanh(  Im[c[0],c[1]-1] /rsig)
            Im[c[0]+1,c[1]-1] = np.tanh(Im[c[0]+1,c[1]-1]/rsig)
            logger.debug('w %d %d'%c)
            if np.logical_xor(Im[c[0],c[1]-1]>0,Im[c[0]+1,c[1]-1]>0) &(c[1]>1):
                w_expa((c[0],c[1]-1),Im,w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW,di,rsig,I,J)
            if np.logical_xor(Im[c]>0,Im[c[0],c[1]-1]>0) &(c[0]>0):
                n_expa((c[0],c[1]-1),Im,w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW,di,rsig,I,J)
            if np.logical_xor(Im[c[0]+1,c[1]-1]>0,Im[c[0]+1,c[1]]>0) &(c[0]<I-2):
                s_expa((c[0]+1,c[1]-1),Im,w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW,di,rsig,I,J)
            return

        def e_expa(c,Im,w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW,di,rsig,I,J):
            """c: center, top pixel of an edge, tuple"""
            if (abs(Im[c[0],c[1]+1])==1) & (abs(Im[c[0]+1,c[1]+1])==1):
                return
            Im[c[0],c[1]+1]+=(w_W[c[0],c[1]+1]*Im[c]+w_SW[c[0],c[1]+1]*Im[c[0]+1,c[1]])/di[c[0],c[1]+1]
            Im[c[0]+1,c[1]+1]+=(w_NW[c[0]+1,c[1]+1]*Im[c]+w_W[c[0]+1,c[1]+1]*Im[c[0]+1,c[1]]+w_N[c[0]+1,c[1]+1]*Im[c[0],c[1]+1])/di[c[0]+1,c[1]+1]
            Im[c[0],c[1]+1] = np.tanh(  Im[c[0],c[1]+1] /rsig)
            Im[c[0]+1,c[1]+1] = np.tanh(Im[c[0]+1,c[1]+1]/rsig)
            logger.debug('e %d %d'%c)
            if np.logical_xor(Im[c[0],c[1]+1]>0,Im[c[0]+1,c[1]+1]>0) &(c[1]<J-2):
                e_expa((c[0],c[1]+1),Im,w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW,di,rsig,I,J)
            if np.logical_xor(Im[c]>0,Im[c[0],c[1]+1]>0) &(c[0]>0):
                n_expa(c,Im,w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW,di,rsig,I,J)
            if np.logical_xor(Im[c[0]+1,c[1]]>0,Im[c[0]+1,c[1]+1]>0) &(c[0]<I-2):
                s_expa((c[0]+1,c[1]),Im,w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW,di,rsig,I,J)
            return

        # umax=.5
        pim= np.zeros((I+2,J+2))
        pim[floc[0]+1,floc[1]+1] = 1
        pim[gloc[0]+1,gloc[1]+1] = -1
        gc=((gloc[0],floc[0])[gloc[0]>floc[0]],(gloc[1],floc[1])[gloc[1]>floc[1]])
        print('top left loc of gf %d %d'%(gc[0],gc[1]))
        Io = (np.multiply(w_E, pim[1:-1, 2:]) + np.multiply(w_S, pim[2:, 1:-1]) + np.multiply(w_W, pim[1:-1, :J]) \
              + np.multiply( w_N, pim[:I, 1:-1]) + np.multiply(w_SE, pim[2:, 2:]) + np.multiply(w_NE, pim[:I, 2:]) \
              + np.multiply( w_NW, pim[:I, :J]) + np.multiply(w_SW, pim[2:, :J]))/di
        gp = Io + pim[1:-1,1:-1]
        Io= np.zeros((I,J))
        Io[floc[0],floc[1]] = 1
        Io[gloc[0],gloc[1]] = -1
        # im=copy.deepcopy(Io)
        Io[gc[0]:gc[0]+2,gc[1]:gc[1]+2] = np.tanh(gp/rsig)[gc[0]:gc[0]+2,gc[1]:gc[1]+2]
        Io[floc[0],floc[1]] = 1
        Io[gloc[0],gloc[1]] = -1
        ax = plt.subplot(111)
        plt.imshow(Io)
        ax.set_title('pim')
        plt.colorbar(orientation='horizontal')
        plt.show()

        # g f edge expand
        if np.logical_xor(Io[gc]>0,Io[gc[0],gc[1]+1]>0) &(gc[0]>0):
            n_expa(gc,Io,w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW,di,rsig,I,J)
        if np.logical_xor(Io[gc]>0,Io[gc[0]+1,gc[1]]>0) &(gc[1]>0):
            w_expa(gc,Io,w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW,di,rsig,I,J)
        if np.logical_xor(Io[gc[0]+1,gc[1]]>0,Io[gc[0]+1,gc[1]+1]>0) &(gc[0]<I-2):
            s_expa((gc[0]+1,gc[1]),Io,w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW,di,rsig,I,J)
        if np.logical_xor(Io[gc[0],gc[1]+1]>0,Io[gc[0]+1,gc[1]+1]>0) &(gc[1]<J-2):
            e_expa((gc[0],gc[1]+1),Io,w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW,di,rsig,I,J)
        return Io

    # img=ms_seq()
    # return img
    bimg=edge_seq()
    return bimg


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
    # blabels=ig
    # blabels[blabels>0]=1
    # blabels[blabels<0]=-1

    ax = plt.subplot(121)
    # plt.imshow(im[:,:,0])
    plt.imshow(im)
    ax.set_title('sample')
    plt.colorbar(orientation='horizontal')
    ax = plt.subplot(122)
    # plt.imshow(np.square(ig[:,:,0])*np.sum(im))
    plt.imshow(ig)
    # plt.imshow(blabels,cmap='gray')
    ax.set_title('blabel')
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
