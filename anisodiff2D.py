import numpy as np
import matplotlib.image as mpimg
import logging
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
from segment import edge_weight,sparse_w
from scipy.sparse.linalg import eigs
import os

def flux_coeffecient(grad,kappa,option):
    if option==1:
        c=np.exp(-(grad/kappa)**2)
    elif option==2:
        c=1/(1+grad/kappa)
    else:
        logging.error("option needed")
        return
    return c

def anisodiff2D(im,num_iter,delta_t,option,kappaS=.4,kappaE=.4):
    nbin=30
    dy=1
    dx=1
    h,w=im.shape
    for n in range(num_iter):
        pim=np.concatenate((im[:1,:],im,im[-1:,:]),axis=0)
        pim=np.concatenate((pim[:,:1],pim,pim[:,-1:]),axis=1)
        grad_IS= -pim[-1-h:-1,1:1+w]+pim[-h:,1:1+w]
        grad_IE= -pim[1:1+h,-1-w:-1]+pim[1:1+h,-w:]
        grad_IN=np.vstack((np.zeros(w).reshape(1,w),-grad_IS[:-1,:]))
        grad_IW=np.hstack((np.zeros(h).reshape(h,1),-grad_IE[:,:-1]))
        hist,bx=np.histogram(abs(grad_IS).flatten(),bins=nbin)
        kappaS=4*np.dot(hist,bx[:-1])/np.sum(hist)
        hist,bx=np.histogram(abs(grad_IE).flatten(),bins=nbin)
        kappaE=4*np.dot(hist,bx[:-1])/np.sum(hist)
        cS=flux_coeffecient(abs(grad_IS),kappaS,option)
        cE=flux_coeffecient(abs(grad_IE),kappaE,option)
        cN=flux_coeffecient(abs(grad_IN),kappaS,option)
        cW=flux_coeffecient(abs(grad_IW),kappaE,option)
        im=im+delta_t*((1/dy)*(cN*grad_IN+cS*grad_IS)
                                 +(1/dx)*(cW*grad_IW+cE*grad_IE)
                                 )
        im=im/np.sum(im)
        # print('iter %d, kappa %.2f %.2f'%(n,kappaE,kappaS))
    # print(pbS)
    return im

if __name__=="__main__":
    # with open('imgs/tri_part','rb') as f:
    #     ims=np.load(f)
    # im=ims
    data_path=os.path.join( os.getcwd(),'photos')
    im_flist=os.listdir(data_path)
    im_no=1
    ims=mpimg.imread(os.path.join(data_path,im_flist[im_no]))
    im=ims[100:300,:300,2]
    im=im/np.sum(im)

    num_iter=100
    option=1
    delta_t=1/7
    ad=anisodiff2D(im,num_iter,delta_t,option)

    # ax=plt.subplot(121)
    # plt.imshow(im,cmap='Blues')
    # plt.colorbar(orientation='horizontal')
    # ax=plt.subplot(122)
    # plt.imshow(ad,cmap='Blues')
    # plt.colorbar(orientation='horizontal')
    # plt.show()
    # with open('test/%s_ad_iter50_1'%im_flist[im_no],'wb') as f:
    #     np.save(f,ad)
    # with open('test/%s_ad_iter50_1'%im_flist[im_no],'rb') as f:
    #     ad=np.load(f)

    h,w=im.shape
    pim=np.concatenate((ad[:1,:],ad,ad[-1:,:]),axis=0)
    pim=np.concatenate((pim[:,:1],pim,pim[:,-1:]),axis=1)
    grad_IS= -pim[-1-h:-1,1:1+w]+pim[-h:,1:1+w]
    grad_IE= -pim[1:1+h,-1-w:-1]+pim[1:1+h,-w:]
    grad_IN=np.vstack((np.zeros(w).reshape(1,w),-grad_IS[:-1,:]))
    grad_IW=np.hstack((np.zeros(h).reshape(h,1),-grad_IE[:,:-1]))
    I,J=im.shape
    N=I*J
    l=np.arange(N).reshape(I,J)

    # laplacian matrix of edge pixels
    nbin=30
    hist,bx=np.histogram(abs(grad_IS).flatten(),bins=nbin)
    gradm2=10*np.dot(hist,bx[:-1])/np.sum(hist)
    hist,bx=np.histogram(abs(grad_IE).flatten(),bins=nbin)
    gradm1=10*np.dot(hist,bx[:-1])/np.sum(hist)
    gm2=abs(grad_IS)>gradm2
    gm1=abs(grad_IE)>gradm1
    # 1,2: E,S
    w1i=l[gm1]
    w1si=l[:-1,][gm1[:-1,]]
    w1di=l[1:,1:][gm1[:-1,:-1]]
    w1e=np.zeros(w1i.shape[0])
    w1s=(abs(grad_IS)[:-1,:][gm1[:-1,:]]<gradm1).astype('int')
    w1n=(abs(grad_IN)[1:,1:][gm1[:-1,:-1]]<gradm1).astype('int')
    w1w=(abs(grad_IW)[1:,1:][gm1[:-1,:-1]]<gradm1).astype('int')
    w2i=l[gm2]
    w2ei=l[:,:-1][gm2[:,:-1]]
    w2di=l[1:,1:][gm2[:-1,:-1]]
    w2e=(abs(grad_IE)[:,:-1][gm2[:,:-1]]<gradm2).astype('int')
    w2s=np.zeros(w2i.shape[0])
    w2n=(abs(grad_IN)[1:,1:][gm2[:-1,:-1]]<gradm2).astype('int')
    w2w=(abs(grad_IW)[1:,1:][gm2[:-1,:-1]]<gradm2).astype('int')
    gm1n=np.hstack((np.zeros((I,1)),gm1[:,:-1])).astype('bool')
    gm2n=np.vstack((np.zeros((1,J)),gm2[:-1,])).astype('bool')
    w1nni=l[1:,:][gm1n[1:,:]]
    w1ndi=l[:-1,:-1][gm1n[1:,1:]]
    w1nn=(abs(grad_IN)[1:,:][gm1n[1:,:]]<gradm1).astype('int')
    w1ne=(abs(grad_IE)[:-1,:-1][gm1n[1:,1:]]<gradm1).astype('int')
    w1ns=(abs(grad_IS)[:-1,:-1][gm1n[1:,1:]]<gradm1).astype('int')
    w2nwi=l[:,1:][gm2n[:,1:]]
    w2ndi=l[:-1,:-1][gm2n[1:,1:]]
    w2nw=(abs(grad_IW)[:,1:][gm2n[:,1:]]<gradm2).astype('int')
    w2ne=(abs(grad_IE)[:-1,:-1][gm2n[1:,1:]]<gradm2).astype('int')
    w2ns=(abs(grad_IS)[:-1,:-1][gm2n[1:,1:]]<gradm2).astype('int')

    wi=np.concatenate((w1i,w1si,w1di,w1di,w2ei,w2i,w2di,w2di,w1nni,w1ndi,w1ndi,w2nwi,w2ndi,w2ndi))
    ax=plt.subplot(131)
    plt.imshow(ad,cmap='gray')
    plt.colorbar(orientation='horizontal')
    ax=plt.subplot(132)
    plt.imshow(im,cmap='gray')
    plt.colorbar(orientation='horizontal')
    ax=plt.subplot(133)
    plt.imshow(((gm1.astype('int')+gm2.astype('int')+gm1n+gm2n)>0).astype('int'),cmap='gray')
    ax.set_title('edge pixels')
    plt.colorbar(orientation='horizontal')
    plt.show()
    import sys
    sys.exit()
    wj=np.concatenate((w1i+1,w1si+J,w1di-J,w1di-1,w2ei+1,w2i+J,w2di-J,w2di-1,w1nni-J,w1ndi+1,w1ndi+J,w2nwi-1,w2ndi+1,w2ndi+J))
    we=np.concatenate((w1e,w1s,w1n,w1w,w2e,w2s,w2n,w2w,w1nn,w1ne,w1ns,w2nw,w2ne,w2ns))
    W=csr_matrix((we,(wi,wj)),shape=(N,N)).toarray()
    W=W+W.T
    row0=np.sum(abs(W),1)
    edge_idx=np.arange(N)[row0!=0]
    W=W[edge_idx,:][:,edge_idx]
    d=np.sum(W,axis=1)
    D=np.diag(d)
    L=D-W

    L=L.astype('float')
    D=D.astype('float')
    w,v=eigs(L,k=2,M=D,which='SM')
    with open('test/tri_edge_SM_seg1','wb') as f:
        np.save(f,w)
        np.save(f,v)
    # with open('test/tri_edge_SM_seg1','rb') as f:
    #     w=np.load(f)
    #     v= np.load(f)
    sm_idx= int(w[0] < w[1])
    print(sm_idx,w)
    e1=v[:,sm_idx].astype(np.float)
    seg1=e1>0

    le=np.zeros(N)
    le[edge_idx[seg1]]=1
    le[edge_idx[np.logical_not(seg1)]]=2
    ax=plt.subplot(2,3,3)
    plt.imshow(le.reshape(I,J),cmap='gray')
    ax.set_title('w:%f, seg1'%w[sm_idx])
    plt.colorbar(orientation='horizontal')

    sub_idx=edge_idx[seg1]
    eloc=np.arange(len(edge_idx))[seg1]
    L1=L[eloc,:][:,eloc]
    w,v=eigs(L1,k=2,M=np.diag(np.diag(L1)),which='SM')
    sm_idx= int(w[0] < w[1])
    print(sm_idx,w)
    e1=v[:,sm_idx].astype(np.float)
    seg11=e1>0
    le[sub_idx[seg11]]=3
    le[sub_idx[np.logical_not(seg11)]]=4
    ax=plt.subplot(2,3,4)
    plt.imshow(le.reshape(I,J),cmap='gray')
    ax.set_title('w:%f, seg11'%w[sm_idx])
    plt.colorbar(orientation='horizontal')

    seg1=np.logical_not(seg1)
    sub_idx=edge_idx[seg1]
    eloc=np.arange(len(edge_idx))[seg1]
    L1=L[eloc,:][:,eloc]
    w,v=eigs(L1,k=2,M=np.diag(np.diag(L1)),which='SM')
    # w,v=eigs(L1,k=2,which='SM')
    sm_idx= int(w[0] < w[1])
    print(sm_idx,w)
    e1=v[:,sm_idx].astype(np.float)
    seg1=e1>0
    le[sub_idx[seg1]]=5
    le[sub_idx[np.logical_not(seg1)]]=6
    ax=plt.subplot(2,3,5)
    plt.imshow(le.reshape(I,J),cmap='gray')
    ax.set_title('w:%f, seg01'%w[sm_idx])
    plt.colorbar(orientation='horizontal')

    ax=plt.subplot(236)
    plt.imshow(le.reshape(I,J),cmap='gray')
    ax.set_title('edge seg')
    plt.colorbar(orientation='horizontal')
    ax=plt.subplot(232)
    plt.imshow(ad,cmap='gray')
    ax.set_title('smooth')
    plt.colorbar(orientation='horizontal')
    ax=plt.subplot(231)
    plt.imshow(im,cmap='gray')
    ax.set_title('sample')
    plt.colorbar(orientation='horizontal')
    plt.show()
