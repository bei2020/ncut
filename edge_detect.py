# coding: utf-8
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import os
import copy


def conv_gaussian_lap(sig,img):
    """Return laplacian of gaussian convoled image"""
    Glap = lambda x, sig: -1 / (np.pi * sig ** 4) * (1 - x ** 2 / (2 * sig ** 2)) * np.exp(-x ** 2 / (2 * sig ** 2))
    # Glap = lambda x, sig: -1 * (1 - x ** 2 / (2 * sig ** 2)) * np.exp(-x ** 2 / (2 * sig ** 2))
    # 1234:ESWN
    sigr = 3
    x = np.arange(sig * sigr + 1)
    Gx = Glap(x, sig)
    # interval  * four directions
    Gs = np.outer(Gx[1:], np.ones(4))
    Io = np.stack( (np.concatenate((im[:, 1:], np.zeros(I).reshape(I, 1)), 1),
                    np.concatenate((im[1:, :], np.zeros(J).reshape(1, J))),
                    np.concatenate((np.zeros(I).reshape(I,1),im[:,:-1]),1),
                    np.concatenate((np.zeros(J).reshape(1,J),im[:-1,:])),
                    ))
    for s in x[2:]:
        Io=np.vstack( (Io,
                       np.concatenate((im[:, s:], np.zeros(I*s).reshape(I, s)), 1).reshape(1,I,J),
                       np.concatenate((im[s:, :], np.zeros(J*s).reshape(s, J))).reshape(1,I,J),
                       np.concatenate((np.zeros(I*s).reshape(I,s),im[:,:-s]),1).reshape(1,I,J),
                       np.concatenate((np.zeros(J*s).reshape(s,J),im[:-s,:])).reshape(1,I,J),
                        )
                       )
    Isig = np.sum(np.multiply(Gs.flatten().reshape(Io.shape[0],1,1),Io),0)
    Isig+=im*Gx[0]
    return Isig

def conv_gaussian(sig,img):
    """Return gaussian convoled image"""
    I,J=img.shape
    Ga = lambda x, sig: 1 / (2*np.pi * sig ** 2) * np.exp(-x ** 2 / (2*np.pi * sig ** 2))
    # 1234:ESWN
    sigr = 3
    x = np.arange(sig * sigr + 1).astype('int')
    Gx = Ga(x, sig)
    # interval  * four directions
    Gs = np.outer(Gx[1:], np.ones(4))
    Io = np.stack( (np.concatenate((img[:, 1:], np.zeros(I).reshape(I, 1)), 1),
                    np.concatenate((img[1:, :], np.zeros(J).reshape(1, J))),
                    np.concatenate((np.zeros(I).reshape(I,1),img[:,:-1]),1),
                    np.concatenate((np.zeros(J).reshape(1,J),img[:-1,:])),
                    ))
    for s in x[2:]:
        Io=np.vstack( (Io,
                       np.concatenate((img[:, s:], np.zeros(I*s).reshape(I, s)), 1).reshape(1,I,J),
                       np.concatenate((img[s:, :], np.zeros(J*s).reshape(s, J))).reshape(1,I,J),
                       np.concatenate((np.zeros(I*s).reshape(I,s),img[:,:-s]),1).reshape(1,I,J),
                       np.concatenate((np.zeros(J*s).reshape(s,J),img[:-s,:])).reshape(1,I,J),
                       )
                      )
    Isig = np.sum(np.multiply(Gs.flatten().reshape(Io.shape[0],1,1),Io),0)
    Isig+=img*Gx[0]
    return Isig

def plot_imgs(imgs):
    """imgs: image list"""
    ni=len(imgs)
    for i in range(ni):
        ax=plt.subplot(2,np.ceil(ni/2).astype('int'),i+1)
        plt.imshow(imgs[i],cmap='Blues')
        ax.set_title('sig:%d'%2**i)
        plt.colorbar(orientation='horizontal')
    plt.show()

def dogs_hasminmax(imgs):
    """Return minmax mask of DoG image in a sequence."""
    maxm=imgs==1
    Dm=copy.deepcopy(maxm)
    # E,S,W,N,NE,ES,WS,WN of each image
    Dm[:,:,1:][maxm[:,:,:-1]]=True
    Dm[:,1:,:][maxm[:,:-1,:]]=True
    Dm[:,:,:-1][maxm[:,:,1:]]=True
    Dm[:,:-1,:][maxm[:,1:,:]]=True
    Dm[:,:-1,1:][maxm[:,1:,:-1]]=True
    Dm[:,1:,1:][maxm[:,:-1,:-1]]=True
    Dm[:,1:,:-1][maxm[:,:-1,1:]]=True
    Dm[:,:-1,:-1][maxm[:,1:,1:]]=True
    # adjcent images
    Dm[1:,:,:][maxm[:-1,:,:]]=True
    Dm[1:,:,1:][maxm[:-1,:,:-1]]=True
    Dm[1:,1:,:][maxm[:-1,:-1,:]]=True
    Dm[1:,:,:-1][maxm[:-1,:,1:]]=True
    Dm[1:,:-1,:][maxm[:-1,1:,:]]=True
    Dm[1:,:-1,1:][maxm[:-1,1:,:-1]]=True
    Dm[1:,1:,1:][maxm[:-1,:-1,:-1]]=True
    Dm[1:,1:,:-1][maxm[:-1,:-1,1:]]=True
    Dm[1:,:-1,:-1][maxm[:-1,1:,1:]]=True

    Dm[:-1,:,:][maxm[1:,:,:]]=True
    Dm[:-1,:,1:][maxm[1:,:,:-1]]=True
    Dm[:-1,1:,:][maxm[1:,:-1,:]]=True
    Dm[:-1,:,:-1][maxm[1:,:,1:]]=True
    Dm[:-1,:-1,:][maxm[1:,1:,:]]=True
    Dm[:-1,:-1,1:][maxm[1:,1:,:-1]]=True
    Dm[:-1,1:,1:][maxm[1:,:-1,:-1]]=True
    Dm[:-1,1:,:-1][maxm[1:,:-1,1:]]=True
    Dm[:-1,:-1,:-1][maxm[1:,1:,1:]]=True

    minm=imgs==-1
    Dmi=copy.deepcopy(minm)
    # E,S,W,N,NE,ES,WS,WN
    Dmi[:,:,1:][minm[:,:,:-1]]=True
    Dmi[:,1:,:][minm[:,:-1,:]]=True
    Dmi[:,:,:-1][minm[:,:,1:]]=True
    Dmi[:,:-1,:][minm[:,1:,:]]=True
    Dmi[:,:-1,1:][minm[:,1:,:-1]]=True
    Dmi[:,1:,1:][minm[:,:-1,:-1]]=True
    Dmi[:,1:,:-1][minm[:,:-1,1:]]=True
    Dmi[:,:-1,:-1][minm[:,1:,1:]]=True
    #adjcent
    Dmi[1:,:,:][minm[:-1,:,:]]=True
    Dmi[1:,:,1:][minm[:-1,:,:-1]]=True
    Dmi[1:,1:,:][minm[:-1,:-1,:]]=True
    Dmi[1:,:,:-1][minm[:-1,:,1:]]=True
    Dmi[1:,:-1,:][minm[:-1,1:,:]]=True
    Dmi[1:,:-1,1:][minm[:-1,1:,:-1]]=True
    Dmi[1:,1:,1:][minm[:-1,:-1,:-1]]=True
    Dmi[1:,1:,:-1][minm[:-1,:-1,1:]]=True
    Dmi[1:,:-1,:-1][minm[:-1,1:,1:]]=True

    Dmi[:-1,:,:][minm[1:,:,:]]=True
    Dmi[:-1,:,1:][minm[1:,:,:-1]]=True
    Dmi[:-1,1:,:][minm[1:,:-1,:]]=True
    Dmi[:-1,:,:-1][minm[1:,:,1:]]=True
    Dmi[:-1,:-1,:][minm[1:,1:,:]]=True
    Dmi[:-1,:-1,1:][minm[1:,1:,:-1]]=True
    Dmi[:-1,1:,1:][minm[1:,:-1,:-1]]=True
    Dmi[:-1,1:,:-1][minm[1:,:-1,1:]]=True
    Dmi[:-1,:-1,:-1][minm[1:,1:,1:]]=True
    return np.logical_and(Dm,Dmi)

def dog_hasminmax(img):
    """Return minmax mask of a DoG image."""
    maxm=img==1
    Dm=copy.deepcopy(maxm)
    # E,S,W,N,NE,ES,WS,WN
    Dm[:,1:][maxm[:,:-1]]=True
    Dm[1:,:][maxm[:-1,:]]=True
    Dm[:,:-1][maxm[:,1:]]=True
    Dm[:-1,:][maxm[1:,:]]=True
    Dm[:-1,1:][maxm[1:,:-1]]=True
    Dm[1:,1:][maxm[:-1,:-1]]=True
    Dm[1:,:-1][maxm[:-1,1:]]=True
    Dm[:-1,:-1][maxm[1:,1:]]=True

    minm=img==-1
    Dmi=copy.deepcopy(minm)
    # E,S,W,N,NE,ES,WS,WN
    Dmi[:,1:][minm[:,:-1]]=True
    Dmi[1:,:][minm[:-1,:]]=True
    Dmi[:,:-1][minm[:,1:]]=True
    Dmi[:-1,:][minm[1:,:]]=True
    Dmi[:-1,1:][minm[1:,:-1]]=True
    Dmi[1:,1:][minm[:-1,:-1]]=True
    Dmi[1:,:-1][minm[:-1,1:]]=True
    Dmi[:-1,:-1][minm[1:,1:]]=True

    # Dm[:,2:][maxm[:,:-2]]=True
    # Dm[2:,:][maxm[:-2,:]]=True
    # Dm[:,:-2][maxm[:,2:]]=True
    # Dm[:-2,:][maxm[2:,:]]=True
    # Dm[:-2,2:][maxm[2:,:-2]]=True
    # Dm[2:,2:][maxm[:-2,:-2]]=True
    # Dm[2:,:-2][maxm[:-2,2:]]=True
    # Dm[:-2,:-2][maxm[2:,2:]]=True

    # Dmi[:,2:][minm[:,:-2]]=True
    # Dmi[2:,:][minm[:-2,:]]=True
    # Dmi[:,:-2][minm[:,2:]]=True
    # Dmi[:-2,:][minm[2:,:]]=True
    # Dmi[:-2,2:][minm[2:,:-2]]=True
    # Dmi[2:,2:][minm[:-2,:-2]]=True
    # Dmi[2:,:-2][minm[:-2,2:]]=True
    # Dmi[:-2,:-2][minm[2:,2:]]=True
    return np.logical_and(Dm,Dmi)

def sps(imgs):
    """Return threshold of minmax in a DoG sequence."""
    hbl=[np.histogram(D[i,:,:].flatten(),bins=nbin) for i in range(D.shape[0])]
    sp=[np.dot(hb[0],hb[1][:-1])/np.sum(hb[0]) for hb in hbl]
    return sp


if __name__=="__main__":
    data_path = os.path.join(os.getcwd(), 'photos')
    im_flist = os.listdir(data_path)
    im_no = 1
    ims = mpimg.imread(os.path.join(data_path, im_flist[im_no]))
    # im=ims[:,:,2]
    im = ims[:, :, 2][:1000, :1000]

    im=im/np.max(im)
    I, J = im.shape

    nint=2
    nbin=30
    no=2
    # k=2^[-1,0,1,2,3]/s, s=nint
    sig=[2**(i/nint) for i in range(-1,nint+2)]
    Lsig=np.array([conv_gaussian(sig[i],im) for i in range(len(sig))])
    Lm=np.max(Lsig,(1,2)).reshape((Lsig.shape[0],1,1))
    Lsig=np.multiply(Lsig,1/Lm)
    D=Lsig[1:,:,:]-Lsig[:1,:,:]
    # hist,bx=np.histogram(D[0,:,:].flatten(),bins=nbin)
    # sp=np.dot(hist,bx[:-1])/np.sum(hist)
    # Dq=[copy.deepcopy(D[0,:,:])]
    # Dq[-1][abs(Dq[-1])<abs(sp)]=0
    # Dq[-1][Dq[-1]>0]=1
    # Dq[-1][Dq[-1]<0]=-1
    Dd=[D[0,:,:]]

    for o in range(1,no):
        Lsig=Lsig[-2:,:,:][:,::2,::2]
        Lsig=np.vstack((conv_gaussian(sig[0],Lsig[0,:,:]).reshape((1,Lsig.shape[1],Lsig.shape[2])),
                        Lsig,np.array([conv_gaussian(s,Lsig[0,:,:]) for s in sig[3:]])))
        Lm=np.max(Lsig,(1,2)).reshape((Lsig.shape[0],1,1))
        Lsig=np.multiply(Lsig,1/Lm)
        D=Lsig[1:,:,:]-Lsig[:1,:,:]
        sp=np.array(sps(D))
        Dq=copy.deepcopy(D)
        Dq[abs(Dq)<abs(sp.reshape((D.shape[0],1,1)))]=0
        Dq[Dq>0]=1
        Dq[Dq<0]=-1
        Dd=[*Dd,D[0,:,:]]

        Dm=dog_hasminmax(Dq[0])
        Dqm=copy.deepcopy(Dq[0])
        Dqm[np.logical_not(Dm)]=0
        ax=plt.subplot(121)
        # plt.imshow(Dd[1],cmap='Blues')
        plt.imshow(Dqm,cmap='Blues')
        ax.set_title('edge pixels, sigma=2')
        plt.colorbar(orientation='horizontal')
        ax=plt.subplot(122)
        plt.imshow(ims[:1000, :1000],cmap='Blues')
        ax.set_title('sample')
        plt.colorbar(orientation='horizontal')
        plt.show()

    # plot_imgs(Dq)
    # plot_imgs(D)
