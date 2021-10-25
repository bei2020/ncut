import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix
from lanc_eig import eiges
import os
import matplotlib.image as mpimg
from PIL import Image

def plot_eigs(w,v,shape=None,row=2):
    if shape==None:
        shape=(int(v.shape[0]**(1/2)),int(v.shape[0]**(1/2)))
    # nump=min(len(w),7)
    for i in range(len(w)):
        ax=plt.subplot(row,4,i+1)
        plt.imshow(v[:,i].reshape(shape))
        ax.set_title('w:%f'%w[i])
        plt.colorbar(orientation='horizontal')

def sparse_w(mtrx, emin=1e-6):
    """Return sparse matrix with element 0 below threshold."""
    return  np.multiply(mtrx>emin,mtrx)

def edge_weight(intensity_diff,sig=0.2):
    return np.exp(-np.sum(intensity_diff**2,-1)/sig**2)

def edge_weight_g(intensity_diff,sig=0.2):
    return np.exp(-intensity_diff**2/sig**2)

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

def img_lap(img,rsig=.1):
    #123: E,SE,S
    if len(img.shape)==2:
        I,J=img.shape
        N=I*J
        g1=img[:,:-1]-img[:,1:]
        g2=img[:-1,:-1]-img[1:,1:]
        g3=img[:-1,:]-img[1:,]
        rsig = grad_m(np.concatenate((g1.flatten(),g2.flatten(),g3.flatten())))
        print('rsig %f'%rsig)
        w1=edge_weight_g(g1,rsig)
        w2=edge_weight_g(g2,rsig)
        w3=edge_weight_g(g3,rsig)
    else:
        I,J,K=img.shape
        N=I*J
        g1=img[:,:-1]-img[:,1:]
        g2=img[:-1,:-1]-img[1:,1:]
        g3=img[:-1,:]-img[1:,]
        rsig = grad_m(np.concatenate((g1.flatten(),g2.flatten(),g3.flatten())))
        print('rsig %f'%rsig)
        w1=edge_weight(g1,rsig)
        w2=edge_weight(g2,rsig)
        w3=edge_weight(g3,rsig)

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

    W=sparse_w(W+W.T,1e-16)
    W/=np.sum(W)
    dmm=np.sum(W,axis=0)
    D=np.diag(dmm)
    L=D-W
    return L,D

def segimg_lap(seg,img, emin=1e-16):
    """ Return Laplacian of image subgraph.
    seg: segmentation mask of image
    img: intensity image matrix
    """
    # mask of weight matrix 1 2 3
    global I,J,N,l
    w1m=seg[:,1:]&seg[:,:-1]
    w2m=seg[1:,1:]&seg[:-1,:-1]
    w3m=seg[1:,:]&seg[:-1,:]
    w1= np.multiply(w1m,edge_weight_g(img[:,:-1]-img[:,1:]))[w1m]
    w2=np.multiply(w2m, edge_weight_g(img[:-1,:-1]-img[1:,1:]))[w2m]
    w3=np.multiply(w3m, edge_weight_g(img[:-1,:]-img[1:,]))[w3m]
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
    W/=np.sum(W)
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

def sample3part_seg():
    # 3 partition image
    global I,J,N,l
    with open('imgs/tri_part','rb') as f:
        im=np.load(f)
    # ax1=plt.subplot(2,2,1)
    # ax1.imshow(im,cmap='gray')
    ax2=plt.subplot(2,2,1)
    plt.imshow(im,cmap='gray')
    ax2.set_title('centered image')
    plt.colorbar(orientation='horizontal')
    # plt.show()
    I,J=im.shape
    N=I*J
    l=np.arange(N).reshape(im.shape)
    L,D=img_lap(im)
    w,v=eigs(L,k=2,M=D,which='SM')
    # with open('test/tri_centered_SMeigs2_lap3d','wb') as f:
    #     np.save(f,w)
    #     np.save(f,v)
    # with open('test/tri_centered_SMeigs2_lap3d','rb') as f:
    #     w=np.load(f)
    #     v= np.load(f)
    sm_idx= int(w[0] < 1e-15)
    e1=v[:,sm_idx].reshape(I,J).astype(np.float)
    seg1=e1>0
    ax1=plt.subplot(2,2,2)
    plt.imshow(np.multiply(seg1, e1),cmap='gray')
    ax1.set_title('w:%f, seg1'%w[0])
    plt.colorbar(orientation='horizontal')

    # recursive 2 way cut
    L,D=segimg_lap(seg1,im)
    w,v=eigs(L,k=2,M=D,which='SM')
    # with open('test/tri_2nd_SMeigs2_lap2d_seg11','wb') as f:
    #     np.save(f,w)
    #     np.save(f,v)
    # with open('test/tri_2nd_SMeigs2','rb') as f:
    #     w=np.load(f)
    #     v= np.load(f)
    e11=np.zeros(N)
    sm_idx= int(w[0] < 1e-15)
    e11[l[seg1]]=v[:,sm_idx].astype(np.float)
    e11=e11.reshape(I,J)
    seg11=e11>0
    # ax1=plt.subplot(1,2,1)
    # plt.imshow(e11,cmap='gray')
    # plt.colorbar(orientation='horizontal')
    ax2=plt.subplot(2,2,3)
    plt.imshow(np.multiply(seg11,e11),cmap='gray')
    ax2.set_title('w:%f, seg11'%w[0])
    plt.colorbar(orientation='horizontal')
    # plt.show()

    seg1=np.logical_not(seg1)
    L,D=segimg_lap(seg1,im)
    w,v=eigs(L,k=2,M=D,which='SM')
    # with open('test/tri_2ndn_SMeigs2_lap2d_seg01','wb') as f:
    #     np.save(f,w)
    #     np.save(f,v)
    # with open('test/tri_2ndn_SMeigs2','rb') as f:
    #     w=np.load(f)
    #     v= np.load(f)
    e11=np.zeros(N)
    sm_idx= int(w[0] < 1e-15)
    e11[l[seg1]]=v[:,sm_idx].astype(np.float)
    e11=e11.reshape(I,J)
    seg11=e11>0
    ax2=plt.subplot(2,2,4)
    plt.imshow(np.multiply(seg11,e11),cmap='gray')
    ax2.set_title('w:%f, seg01'%w[0])
    plt.colorbar(orientation='horizontal')
    plt.show()
    #check image value is continuous, stop cut

if __name__=="__main__":
    # 2 partition image
    # sample2part_seg()
    # # 3 partition image
    # sample3part_seg()

    data_path = os.path.join(os.getcwd(), 'photos')
    im_flist = os.listdir(data_path)
    im_no = 3
    img = mpimg.imread(os.path.join(data_path, im_flist[im_no]))
    img = img[40:60, 10:50, :]
    # im = im[10:22, 10:50,:]
    # # im=im[110:150,140:190,:]
    # # im = im[40:50, 10:20, :]
    ime = np.einsum('ijk->k', img.astype('uint32')).reshape(1, 1, img.shape[2])
    im = img / ime

    if len(im.shape)==2:
        I,J=im.shape
    else:
        I,J,K=im.shape
    N=I*J
    l=np.arange(N).reshape(I,J)
    L,D=img_lap(im)
    d = np.diag(D)
    # Ls=np.multiply(np.multiply((d**(-1/2)).reshape(N, 1), L), (d**(-1/2)).reshape(1, N))
    # w,v=eiges(Ls,k=50)

    # with open('test/tri_centered_SMeigs4_lap3d','wb') as f:
    #     np.save(f,w)
    #     np.save(f,v)
    # with open('test/tri_centered_SMeigs5','rb') as f:
    #     w=np.load(f)
    #     v= np.load(f)
    w,v=eigs(L,k=6,M=D,which='SM')
    w=w.astype(np.float)
    v=v.astype(np.float)

    row=2
    plot_eigs(w,v,shape=(I,J),row=row)
    # plot_eigs(w[:11],v[:,:11],shape=(I,J),row=row)
    # plot_eigs(w[:7],v[:,:7],shape=(I,J),row=row)
    ax=plt.subplot(row,4,row*4)
    # plt.imshow(img[10:22, 10:50,:])
    plt.imshow(img)
    ax.set_title('image')
    plt.colorbar(orientation='horizontal')
    plt.show()
