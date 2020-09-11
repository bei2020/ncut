import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from segment import img_lap, segimg_lap
from scipy.sparse.linalg import eigs
import os


if __name__=="__main__":
    data_path=u'photos'
    im_flist=os.listdir(data_path)
    im_no=6
    im=mpimg.imread(os.path.join(data_path,im_flist[0]))

    # down sample rate
    field=4
    # last dimension < 100
    last_map_dim=100
    h=im.shape[0]
    l=int(np.ceil(np.log(h/last_map_dim)/np.log(field)))
    l2=im[:,:,2]
    l2=l2-np.mean(l2)
    # down sample, max pool l times
    # b=0,1...width_down_sample, j_im=4b,4b+2
    for layer in range(l):
        h,wid=l2.shape
        l2=l2[::2,::2][:h//4*2,:wid//4*2]
        h2,w2=l2.shape
        l2=l2.reshape(h2//2,2,w2//2,2)
        l2=np.max(l2,axis=1)
        l2=np.max(l2,axis=2)
        print(l2.shape)
    ax=plt.subplot(1,2,1)
    ax.imshow(im[:,:,2])
    ax.set_title('origin image 3rd channel')
    ax=plt.subplot(1,2,2)
    ax.imshow(l2)
    ax.set_title('down sample rate 4')
    plt.show()

    l2=l2/255
    I,J=l2.shape
    N=I*J
    l=np.arange(N).reshape(l2.shape)
    L,D=img_lap(l2)
    w,v=eigs(L,k=8,M=D,which='SM')
    with open('img%d_centered_d3_sparseidx_eigs8'%im_no,'wb') as f:
        np.save(f,w)
        np.save(f,v)

    # with open('img%d_centered_d3_sparseW_eigs2'%im_no,'rb') as f:
    #     w=np.load(f)
    #     v= np.load(f)

    w=w.astype(np.float)
    v=v.astype(np.float)
    print('2nd SM eig %f'% w[0])
    e1=v[:,0].astype(np.float).reshape(I,J)
    seg1=e1>0
    ax=plt.subplot(2,2,1)
    plt.imshow(np.multiply(seg1,e1))
    plt.colorbar(orientation='horizontal')
    ax.set_title('w:%f, 1st seg>0'%w[0])
    # ax2=plt.subplot(1,2,2)
    # plt.imshow(v1)
    # plt.colorbar(orientation='horizontal')

    # recursive 2 way cut
    depth=3
    for d in range(0,depth):
        L,D=segimg_lap(seg1,l2)
        w,v=eigs(L,k=2,M=D,which='SM')
        with open('img%d_%d_SMeigs2'%(im_no,d+2),'wb') as f:
            np.save(f,w)
            np.save(f,v)
        # with open('img%d_%d_SMeigs2'%(im_no,d+2),'rb') as f:
        #     w=np.load(f)
        #     v= np.load(f)
        sm_idx= int(w[0] < 1e-15)
        e1=np.zeros(N)
        e1[l[seg1]]=v[:,sm_idx].astype(np.float)
        e1=e1.reshape(I,J)
        seg1=e1>0
        # ax1=plt.subplot(1,2,1)
        # plt.imshow(e11)
        # plt.colorbar(orientation='horizontal')
        ax=plt.subplot(2,2,d+2)
        plt.imshow(np.multiply(seg1,e1))
        ax.set_title('w:%f, %d seg>0'%(w[sm_idx],d+2))
        plt.colorbar(orientation='horizontal')
    plt.show()
    im=None

