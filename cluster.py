import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from segment import edge_weight
import copy
import os

def msimg(img,ssig=1,rsig=.1,niter=100):
    """Return image of center values."""
    I,J=img.shape
    sigr = 3
    D=img.shape[2] if len(img.shape)==3 else 1
    a=rsig**2/(D+2)
    x = np.arange(ssig * sigr + 1).astype('int')
    v=np.zeros((img.shape))
    gamma=.9
    # 1234:ESWN
    for i in range(niter):
        Io = np.stack( (np.concatenate(((img[:, 1:]+gamma*v[:,1:]), np.zeros(I).reshape(I, 1)), 1),
                        np.concatenate(((img[1:, :]+gamma*v[1:,:]), np.zeros(J).reshape(1, J))),
                        np.concatenate((np.zeros(I).reshape(I,1),(img[:,:-1])+gamma*v[:,:-1]),1),
                        np.concatenate((np.zeros(J).reshape(1,J),(img[:-1,:]+gamma*v[:-1,:]))),
                        ))
        for s in x[2:]:
            Io=np.vstack( (Io,
                           np.concatenate((img[:, s:]+gamma*v[:,s:], np.zeros(I*s).reshape(I, s)), 1).reshape(1,I,J),
                           np.concatenate((img[s:, :]+gamma*v[s:,:], np.zeros(J*s).reshape(s, J))).reshape(1,I,J),
                           np.concatenate((np.zeros(I*s).reshape(I,s),img[:,:-s]+gamma*v[:,:-s]),1).reshape(1,I,J),
                           np.concatenate((np.zeros(J*s).reshape(s,J),img[:-s,:]+gamma*v[:-s,:])).reshape(1,I,J),
                           )
                          )
        wt=edge_weight(Io-(img.reshape(1,I,J)+gamma*v),rsig)
        Io = np.sum(np.multiply(Io,wt/np.sum(wt,0).reshape((1,I,J))),0)
        v=gamma*v+a*(Io-(img+gamma*v))
        img+=v
    return img

def conti_iter(img,ncont=0,mcont=12,rsig=.1):
    imgc=msimg(copy.deepcopy(img),niter=1)
    if np.array_equal(np.round(np.exp(img),8),np.round(np.exp(imgc),8)):
        print('fix found continue iter %d'%(ncont*1000))
        # print(np.round(np.exp(img), 6))
        # print(np.round(np.exp(imgc), 6))
        return
    else:
        if ncont==mcont:
            print('no fix after continue iter %d'%(ncont*1000))
            print(np.round(np.exp(img), 6))
            print(np.round(np.exp(imgc), 6))
            return
        print('continue')
        ncont+=1
        img=msimg(img,rsig=rsig,niter=1000)
        conti_iter(img,ncont,mcont)

if __name__=="__main__":
    with open('imgs/tri_part','rb') as f:
        im=np.load(f)
    ig=im/np.sum(im)
    ig=np.log(ig)
    # data_path=os.path.join( os.getcwd(),'photos')
    # im_flist=os.listdir(data_path)
    # im_no=1
    # im=mpimg.imread(os.path.join(data_path,im_flist[im_no]))
    # im=im[100:150,50:100,2]
    # ig=im/np.sum(im)
    # ig[ig==0]=.0000001/np.sum(im)
    # ig=np.log(ig)

    h,w=ig.shape
    ig=msimg(ig,niter=1000,rsig=.2)
    conti_iter(ig,mcont=6,rsig=.2)
    labels =(np.round(np.exp(ig)/np.sum(np.exp(ig))*np.sum(im), 1) * 10).astype('int')

    ax=plt.subplot(121)
    plt.imshow(im,cmap='gray')
    ax.set_title('sample')
    plt.colorbar(orientation='horizontal')
    ax=plt.subplot(122)
    plt.imshow(labels,cmap='gray')
    ax.set_title('labels')
    plt.colorbar(orientation='horizontal')
    plt.show()
