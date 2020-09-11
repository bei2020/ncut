import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix

def distance(m,n):
    """Return distance of two pixels m and n base on intensity.
    m,n: both are intensity.
    """
    dis= abs(m-n)
    return dis

def pixel_no(i,j):
    """location in flatten img"""
    return 20*i+j

def sparse_w(mtrx, emin=1e-6):
    """Return sparse matrix with element 0 below threshold."""
    return  np.multiply(mtrx>emin,mtrx)

def edge_weight(intensity_diff,sig=0.2):
    return np.exp(-(intensity_diff)**2/sig**2)

def img_lap(img):
    #123: E,SE,S
    I,J=img.shape
    N=I*J
    w1=edge_weight(img[:,:-1]-img[:,1:])
    w2=edge_weight(img[:-1,:-1]-img[1:,1:])
    w3=edge_weight(img[:-1,:]-img[1:,])

    l=np.arange(I*J).reshape(I,J)
    w1i=l[:,:-1].flatten()
    w1j=w1i+1
    w2i=l[:-1,:-1].flatten()
    w2j=w2i+J+1
    w3i=l[:-1,:].flatten()
    w3j=w3i+J
    wi=np.concatenate((w1i,w2i,w3i))
    wj=np.concatenate((w1j,w2j,w3j))
    we=np.concatenate((w1.flatten(),w2.flatten(),w3.flatten()))
    W=csr_matrix((we,(wi,wj)),shape=(N,N)).toarray()

    W=sparse_w(W+W.T,1e-6)
    dmm=np.sum(W,axis=0)
    D=np.diag(dmm)
    L=D-W
    return L,D

def segimg_lap(seg,img, emin=1e-6):
    """ Return Laplacian of image subgraph.
    seg: segmentation mask of image
    img: intensity image matrix
    """
    # mask of weight matrix 1 2 3
    global I,J,N,l
    w1m=seg[:,1:]&seg[:,:-1]
    w2m=seg[1:,1:]&seg[:-1,:-1]
    w3m=seg[1:,:]&seg[:-1,:]
    w1= np.multiply(w1m,edge_weight(img[:,:-1]-img[:,1:]))[w1m]
    w2=np.multiply(w2m, edge_weight(img[:-1,:-1]-img[1:,1:]))[w2m]
    w3=np.multiply(w3m, edge_weight(img[:-1,:]-img[1:,]))[w3m]
    # w1= edge_weight(img[:,:-1]-img[:,1:])[w1m]
    # w2= edge_weight(img[:-1,:-1]-img[1:,1:])[w2m]
    # w3= edge_weight(img[:-1,:]-img[1:,])[w3m]
    w1=sparse_w(w1,emin)
    w2=sparse_w(w2,emin)
    w3=sparse_w(w3,emin)

    # I,J=img.shape
    # N=I*J
    # l=np.arange(N).reshape(img.shape)
    w1i=l[:,:-1][w1m].flatten()
    w1j=w1i+1
    w2i=l[:-1,:-1][w2m].flatten()
    w2j=w2i+J+1
    w3i=l[:-1,:][w3m].flatten()
    w3j=w3i+J
    wi=np.concatenate((w1i,w2i,w3i))
    wj=np.concatenate((w1j,w2j,w3j))
    we=np.concatenate((w1,w2,w3))
    W=csr_matrix((we,(wi,wj)),shape=(N,N)).toarray()

    W=W+W.T
    sub_idx = l[seg]
    W=W[sub_idx, :][:, sub_idx]
    dmm=np.sum(W,axis=0)
    D=np.diag(dmm)
    L=D-W
    return L,D

def sample2part_seg():
    white=0.2* np.random.randn(20, 5)+1
    white[white>1]=2-white[white>1]
    black=abs(0.2* np.random.randn(20, 15))
    imsh = np.concatenate([white, black], axis=1)

    im=imsh-np.mean(imsh)
    ax1=plt.subplot(1,2,1)
    ax2=plt.subplot(1,2,2)
    ax1.imshow(imsh,cmap='gray')
    ax2.imshow(im,cmap='gray')
    plt.show()
    L,D=img_lap(im)
    w,v=eigs(L,k=3,M=D,which='SM')
    I,J=im.shape
    e1=v[:,0].reshape(I,J)
    e1=e1.astype(np.float)
    e2=v[:,1].reshape(I,J)
    e2=e2.astype(np.float)
    ax=plt.subplot(1,3,1)
    ax.imshow(im,cmap='gray')
    ax.set_title('2 components image')
    ax=plt.subplot(1,3,2)
    ax.imshow(e1,cmap='gray')
    ax.set_title('1st eigen vector, eigen value w:%f'%w[0])
    ax=plt.subplot(1,3,3)
    ax.imshow(e2,cmap='gray')
    ax.set_title('2nd eigen vector, eigen value w:%f'%w[1])
    plt.show()

if __name__=="__main__":
    # 2 partition image
    # sample2part_seg()

    # 3 partition image
    nx,ny=(20,20)
    x = np.linspace(0, 19, nx)
    y = np.linspace(0, 19, ny)
    xv, yv = np.meshgrid(x, y)
    im=np.zeros((20,20))
    im[(xv>=4)&(yv<=10)&(xv*7-4*yv-28>=0)&(xv*3-12*yv+60>=0)]=0.5
    im[(xv>=4)&(yv>=7)&(xv*3-12*yv+60<0)&(xv*13+4*yv-132>=0)]=1
    im[:8,:][im[:8,:]==0]=0
    ms=(im==0.5)
    im[ms]+=np.random.randn(im[ms].shape[0])*0.1
    ms=(im==1)
    im[ms]+=np.random.randn(im[ms].shape[0])*0.1
    ms=(im==0)
    im[ms]+=np.random.randn(im[ms].shape[0])*0.1

    im_c=im-np.mean(im)
    # ax1=plt.subplot(2,2,1)
    # ax1.imshow(im,cmap='gray')
    ax2=plt.subplot(2,2,1)
    plt.imshow(im_c,cmap='gray')
    ax2.set_title('centered image')
    plt.colorbar(orientation='horizontal')
    # plt.show()
    I,J=im_c.shape
    N=I*J
    l=np.arange(N).reshape(im_c.shape)
    L,D=img_lap(im_c)
    w,v=eigs(L,k=2,M=D,which='SM')
    # with open('test/tri_centered_SMeigs2','wb') as f:
    #     np.save(f,w)
    #     np.save(f,v)
    with open('test/tri_centered_SMeigs2','rb') as f:
        w=np.load(f)
        v= np.load(f)
    e1=v[:,0].reshape(I,J).astype(np.float)
    seg1=e1>0
    ax1=plt.subplot(2,2,2)
    plt.imshow(np.multiply(seg1, e1),cmap='gray')
    ax1.set_title('w:%f, 1st seg>0'%w[0])
    plt.colorbar(orientation='horizontal')

    # recursive 2 way cut
    L,D=segimg_lap(seg1,im_c)
    w,v=eigs(L,k=2,M=D,which='SM')
    # with open('test/tri_2nd_SMeigs2','wb') as f:
    #     np.save(f,w)
    #     np.save(f,v)
    with open('test/tri_2nd_SMeigs2','rb') as f:
        w=np.load(f)
        v= np.load(f)
    e11=np.zeros(N)
    e11[l[seg1]]=v[:,0].astype(np.float)
    e11=e11.reshape(I,J)
    seg11=e11>0
    # ax1=plt.subplot(1,2,1)
    # plt.imshow(e11,cmap='gray')
    # plt.colorbar(orientation='horizontal')
    ax2=plt.subplot(2,2,3)
    plt.imshow(np.multiply(seg11,e11),cmap='gray')
    ax2.set_title('w:%f, 2nd seg>0'%w[0])
    plt.colorbar(orientation='horizontal')
    # plt.show()

    seg1=np.logical_not(seg1)
    L,D=segimg_lap(seg1,im_c)
    w,v=eigs(L,k=2,M=D,which='SM')
    # with open('test/tri_2ndn_SMeigs2','wb') as f:
    #     np.save(f,w)
    #     np.save(f,v)
    with open('test/tri_2ndn_SMeigs2','rb') as f:
        w=np.load(f)
        v= np.load(f)
    e11=np.zeros(N)
    e11[l[seg1]]=v[:,0].astype(np.float)
    e11=e11.reshape(I,J)
    seg11=e11>0
    ax2=plt.subplot(2,2,4)
    plt.imshow(np.multiply(seg11,e11),cmap='gray')
    ax2.set_title('w:%f, 2nd seg<0'%w[0])
    plt.colorbar(orientation='horizontal')
    plt.show()
    #check image value is continuous, stop cut
