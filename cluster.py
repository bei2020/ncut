import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from segment import edge_weight
import copy
import os


def msimg(img,ssig=1,rsig=.1,niter=100):
    """Return image of center values."""
    I,J,K=img.shape
    sigr = 3
    D=img.shape[2]
    a=rsig**2/(D+2)
    x = np.arange(ssig * sigr + 1).astype('int')
    v=np.zeros((img.shape))
    gamma=.9
    # 1234:ESWN
    #Io: DHWC
    for i in range(niter):
        Io = np.stack( (np.concatenate(((img[:, 1:,:]+gamma*v[:,1:,:]), np.zeros(I*K).reshape((I,1,K))), 1),
                        np.concatenate(((img[1:, :,:]+gamma*v[1:,:,:]), np.zeros(J*K).reshape((1, J,K)))),
                        np.concatenate((np.zeros(I*K).reshape((I,1,K)),(img[:,:-1,:])+gamma*v[:,:-1,:]),1),
                        np.concatenate((np.zeros(J*K).reshape((1,J,K)),(img[:-1,:,:]+gamma*v[:-1,:,:]))),
                        ))
        for s in x[2:]:
            Io=np.vstack( (Io,
                           np.concatenate((img[:, s:,:]+gamma*v[:,s:,:], np.zeros(I*s*K).reshape((I, s,K))), 1).reshape((1,I,J,K)),
                           np.concatenate((img[s:, :,:]+gamma*v[s:,:,:], np.zeros(J*s*K).reshape((s, J,K)))).reshape((1,I,J,K)),
                           np.concatenate((np.zeros(I*s*K).reshape((I,s,K)),img[:,:-s,:]+gamma*v[:,:-s,:]),1).reshape((1,I,J,K)),
                           np.concatenate((np.zeros(J*s*K).reshape((s,J,K)),img[:-s,:,:]+gamma*v[:-s,:,:])).reshape((1,I,J,K)),
                           )
                          )
        wt=edge_weight(Io-(img+gamma*v).reshape(1,I,J,K),rsig)
        sum0=np.sum(wt,0)
        sum0[sum0==0]=1
        Io = np.sum(np.multiply(Io,(wt/sum0.reshape((1,I,J))).reshape((*wt.shape,1))),0)
        v=gamma*v+a*(Io-(img+gamma*v))
        img+=v
    return img

def conti_iter(img,ncont=0,mcont=12,rsig=.1):
    imgc=msimg(copy.deepcopy(img),niter=1)
    if np.array_equal(np.round(np.exp(img),8),np.round(np.exp(imgc),8)):
        print('fix found continue iter %d'%(ncont*1000))
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
    # with open('imgs/tri_part','rb') as f:
    #     im=np.load(f)
    # ig=im/np.sum(im)
    # ig=np.log(ig).reshape((*im.shape,1))

    data_path=os.path.join( os.getcwd(),'photos')
    im_flist=os.listdir(data_path)
    im_no=1
    im=mpimg.imread(os.path.join(data_path,im_flist[im_no]))
    im=im[100:200,50:300,:]
    ime=np.einsum('ijk->k', im.astype('uint32')).reshape(1,1,im.shape[2])
    ig=im/ime
    ig[ig==0]=.0000001/np.sum(ime)
    ig=np.log(ig)

    ig=msimg(ig,niter=8000,rsig=.05)
    conti_iter(ig,mcont=8,rsig=.08)
    conti_iter(ig,mcont=8,rsig=.1)
    conti_iter(ig,mcont=8,rsig=.15)
    conti_iter(ig,mcont=8,rsig=.2)
    conti_iter(ig,mcont=8,rsig=.3)
    # labels =(np.round(np.exp(ig)/np.sum(np.exp(ig))*np.sum(im), 1) * 10).astype('int')
    # labels=labels[:,:,0]
    labels =(np.round(np.exp(ig)/np.einsum('ijk->k', np.exp(ig)).reshape(1,1,ig.shape[2])*ime, 1)).astype(im.dtype)

    ax=plt.subplot(121)
    # plt.imshow(im[:,:,0])
    plt.imshow(im)
    ax.set_title('sample')
    plt.colorbar(orientation='horizontal')
    ax=plt.subplot(122)
    plt.imshow(labels[:,:,0])
    ax.set_title('.05.08.1.15.2.3  8000')
    plt.colorbar(orientation='horizontal')
    plt.show()
