import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from segment import edge_weight
import copy
import os

def msimg(img,ssig=1,rsig=.1,mcont=5):
    """Return image with center values."""
    I,J,K=img.shape
    D=img.shape[2]
    a=rsig**2/(D+2)
    gamma=.9
    alpha=.99
    lamb=1e-5
    v = np.zeros((img.shape))
    Vp = 0
    img[gloc[0],gloc[1],:]=0
    # 1234:ESWN, Io: DHWC, D=four directions* number of intervals
    # curvature
    Io = np.stack( (np.concatenate((img[:, 1:,:], np.zeros(I*K).reshape((I,1,K))), 1),
                    np.concatenate((img[1:, :,:], np.zeros(J*K).reshape((1, J,K)))),
                    np.concatenate((np.zeros(I*K).reshape((I,1,K)),img[:,:-1,:]),1),
                    np.concatenate((np.zeros(J*K).reshape((1,J,K)),img[:-1,:,:])),
                    np.concatenate((np.concatenate((img[1:,1:,:],np.zeros((I-1,1,K))),1),np.zeros((1,J,K)))),# ES
                    np.concatenate((np.concatenate((np.zeros((I-1,1,K)),img[1:,:-1,:]),1),np.zeros((1,J,K)))),# SW
                    np.concatenate((np.zeros((1,J,K)),np.concatenate((np.zeros((I-1,1,K)),img[:-1,:-1,:]),1))),# NW
                    np.concatenate((np.zeros((1,J,K)),np.concatenate((img[:-1,1:,:],np.zeros((I-1,1,K))),1))),# NE
                    ))
    nint=2
    wt=edge_weight(Io-img.reshape(1,I,J,K),rsig*2**(1/nint))
    wt[:,gloc[0],gloc[1]]=0
    if gloc[1] < J - 1:
        wt[2,gloc[0],gloc[1]+1]=0
    if gloc[0] < I - 1:
        wt[3,gloc[0]+1,gloc[1]]=0
    if gloc[1] > 0:
        wt[0,gloc[0],gloc[1]-1]=0
    if gloc[0] > 0:
        wt[1,gloc[0]-1,gloc[1]]=0
    if gloc[0] < I - 1 & gloc[1] <J-1:
        wt[6,gloc[0]+1,gloc[1]+1]=0
    if gloc[0] < I - 1 & gloc[1] > 0:
        wt[7,gloc[0]+1,gloc[1]-1]=0
    if gloc[0] > 0 & gloc[1] > 0:
        wt[4,gloc[0]-1,gloc[1]-1]=0
    if gloc[0] > 0 & gloc[1] < J - 1:
        wt[5,gloc[0]-1,gloc[1]+1]=0
    sum0=np.sum(wt,0)
    sum0[sum0==0]=1
    Iok = np.sum(np.multiply(Io,wt.reshape((*wt.shape,1)))/sum0.reshape((I,J,1)),0)
    fi = np.zeros((Iok.shape))
    fi[floc[0],floc[1],:] = 1
    Iok += fi
    wt=edge_weight(Io-img.reshape(1,I,J,K),rsig)
    wt[:,gloc[0],gloc[1]]=0
    if gloc[1] < J - 1:
        wt[2,gloc[0],gloc[1]+1]=0
    if gloc[0] < I - 1:
        wt[3,gloc[0]+1,gloc[1]]=0
    if gloc[1] > 0:
        wt[0,gloc[0],gloc[1]-1]=0
    if gloc[0] > 0:
        wt[1,gloc[0]-1,gloc[1]]=0
    if gloc[0] < I - 1 & gloc[1] <J-1:
        wt[6,gloc[0]+1,gloc[1]+1]=0
    if gloc[0] < I - 1 & gloc[1] > 0:
        wt[7,gloc[0]+1,gloc[1]-1]=0
    if gloc[0] > 0 & gloc[1] > 0:
        wt[4,gloc[0]-1,gloc[1]-1]=0
    if gloc[0] > 0 & gloc[1] < J - 1:
        wt[5,gloc[0]-1,gloc[1]+1]=0
    sum0=np.sum(wt,0)
    sum0[sum0==0]=1
    Io = np.sum(np.multiply(Io,wt.reshape((*wt.shape,1)))/sum0.reshape((I,J,1)),0)
    Io += fi
    cur=Iok-Io
    gp=Io-img
    Vp=alpha*Vp+(1-alpha)*gp**2
    Gp=1/(lamb+np.sqrt(Vp))
    v=gamma*v+a*(np.multiply(Gp,gp)+cur)
    img+=v
    def msiter(niter=1000):
        nonlocal img,v,Vp,wt,cur
        for i in range(niter):
            Io = np.stack( (np.concatenate((img[:, 1:,:]+gamma*v[:,1:,:], np.zeros(I*K).reshape((I,1,K))), 1),
                            np.concatenate((img[1:, :,:]+gamma*v[1:,:,:], np.zeros(J*K).reshape((1, J,K)))),
                            np.concatenate((np.zeros(I*K).reshape((I,1,K)),img[:,:-1,:]+gamma*v[:,:-1,:]),1),
                            np.concatenate((np.zeros(J*K).reshape((1,J,K)),img[:-1,:,:]+gamma*v[:-1,:,:])),
                            np.concatenate((np.concatenate((img[1:,1:,:]+gamma*v[1:,1:,:],np.zeros((I-1,1,K))),1),np.zeros((1,J,K)))),# ES
                            np.concatenate((np.concatenate((np.zeros((I-1,1,K)),img[1:,:-1,:]+gamma*v[1:,:-1,:]),1),np.zeros((1,J,K)))),# SW
                            np.concatenate((np.zeros((1,J,K)),np.concatenate((np.zeros((I-1,1,K)),img[:-1,:-1,:]+gamma*v[:-1,:-1,:]),1))),# NW
                            np.concatenate((np.zeros((1,J,K)),np.concatenate((img[:-1,1:,:]+gamma*v[:-1,1:,:],np.zeros((I-1,1,K))),1))),# NE
                            ))
            Io = np.sum(np.multiply(Io,wt.reshape((*wt.shape,1)))/sum0.reshape((I,J,1)),0)
            Io+=fi
            gp=Io-(img+gamma*v)
            Vp=alpha*Vp+(1-alpha)*gp**2
            Gp=1/(lamb+np.sqrt(Vp))
            v=gamma*v+a*(np.multiply(Gp,gp)+cur)
            img+=v
    def msconti(ncont=0,mcont=12):
        nonlocal img
        imgc=copy.deepcopy(img)
        msiter(10)
        if np.array_equal(np.round(np.exp(img),8),np.round(np.exp(imgc),8)):
            # if np.array_equal(np.round(img,8),np.round(imgc,8)):
            print('fix found continue iter %d'%(ncont*1000))
            return
        else:
            if ncont==mcont:
                print('no fix after continue iter %d'%(ncont*1000))
                # print(np.round(img[:10,:10,0], 6))
                # print(np.round(imgc[:10,:10,0], 6))
                print(np.round(np.exp(img[:10,:10,0]), 6))
                print(np.round(np.exp(imgc[:10,:10,0]), 6))
                return
            print('continue')
            ncont+=1
            msiter(niter=1000)
            msconti(ncont,mcont)
    msiter()
    msconti(mcont=mcont)
    img[gloc[0],gloc[1],:]=-np.inf
    return img

if __name__=="__main__":
    # with open('imgs/tri_part','rb') as f:
    #     im=np.load(f)
    # iph=im/np.sum(im)
    # iph=np.log(iph).reshape((*im.shape,1))

    data_path=os.path.join( os.getcwd(),'photos')
    im_flist=os.listdir(data_path)
    im_no=0
    im=mpimg.imread(os.path.join(data_path,im_flist[im_no]))
    # im=im[110:150,140:190,:]
    im=im[40:60,10:50,:]
    ime=np.einsum('ijk->k', im.astype('uint32')).reshape(1,1,im.shape[2])
    iph=im/ime
    iph[iph==0]=.0000001/np.sum(ime)
    iph=np.log(iph)

    I,J,K=iph.shape
    gloc=[0,0]
    l=np.arange(I*J).reshape(I,J)
    iph[gloc[0],gloc[1],:]=-np.inf
    fidx=np.stack([l]*K,axis=2)[iph==np.max(iph)][0]
    floc=[fidx//J,fidx%J]
    ig=msimg(copy.deepcopy(iph),rsig=.08,mcont=6)

    blabels=np.zeros((I,J,6))
    mct=np.einsum('ijk->k',ig==np.max(ig,-1).reshape(I,J,1).astype('int'))
    ch=np.arange(K)[mct==np.max(mct)][0]
    prob = 1 / (1 + np.exp(-ig[:,:,ch]))
    blabels[:,:,0] = (prob > np.random.rand(*prob.shape)).astype('int')

    ax=plt.subplot(231)
    plt.imshow(im)
    ax.set_title('sample')
    plt.colorbar(orientation='horizontal')
    ax=plt.subplot(232)
    plt.imshow(ig[:,:,ch])
    ax.set_title('iph1')
    plt.colorbar(orientation='horizontal')
    ax=plt.subplot(233)
    plt.imshow(blabels[:,:,0])
    ax.set_title('seg1 .08 6000')
    plt.colorbar(orientation='horizontal')

    cmsk=np.sum(blabels,-1).astype('bool') # cut nodes mask
    fidx=np.stack([l]*K,axis=2)[im==np.max(im[np.logical_not(cmsk)])][0]
    floc=[fidx//J,fidx%J]
    if floc[0]!=I-1 & floc[0]!=0 & floc[1]!=J-1 & floc[1]!=0:
        sqm=np.logical_not(cmsk)[floc[0]-1:floc[0]+2,floc[1]-1:floc[1]+2].flatten().tolist()
        sqm=np.array([*sqm[:4],*sqm[5:]])
        sql=np.array([(floc[0]-1,floc[1]-1),(floc[0]-1,floc[1]),(floc[0]-1,floc[1]+1),(floc[0],floc[1]-1),(floc[0],floc[1]+1),(floc[0]+1,floc[1]-1),(floc[0]+1,floc[1]),(floc[0]+1,floc[1]+1)])[sqm]
        sq=iph[:,:,ch][floc[0]-1:floc[0]+2,floc[1]-1:floc[1]+2]
        sq=sq.flatten().tolist()
        sq=np.array([*sq[:4],*sq[5:]])[sqm]
        sq=sq<sq[0]
        if np.sum(sq.astype('int'))==0:
            floc=sql[1]
        else:
            floc=sql[sq][0]
    iph[cmsk] = 0
    ig=msimg(copy.deepcopy(iph),rsig=.1,mcont=6)
    ig[cmsk] = -np.inf
    mct=np.einsum('ijk->k',ig==np.max(ig,-1).reshape(I,J,1).astype('int'))
    ch=np.arange(K)[mct==np.max(mct)][0]
    prob = 1 / (1 + np.exp(-ig[:,:,ch]))
    blabels[:,:,1] = (prob > np.random.rand(*prob.shape)).astype('int')

    ax=plt.subplot(234)
    plt.imshow(ig[:,:,0])
    ax.set_title('iph01')
    plt.colorbar(orientation='horizontal')
    ax=plt.subplot(235)
    plt.imshow(blabels[:,:,1])
    ax.set_title('seg01 .1 6000')
    plt.colorbar(orientation='horizontal')

    cmsk=np.sum(blabels,-1).astype('bool')
    fidx=np.stack([l]*K,axis=2)[im==np.max(im[np.logical_not(cmsk)])][0]
    floc=(fidx//J,fidx%J)
    if floc[0]!=I-1 & floc[0]!=0 & floc[1]!=J-1 & floc[1]!=0:
        sqm=np.logical_not(cmsk)[floc[0]-1:floc[0]+2,floc[1]-1:floc[1]+2].flatten().tolist()
        sqm=np.array([*sqm[:4],*sqm[5:]])
        sql=np.array([(floc[0]-1,floc[1]-1),(floc[0]-1,floc[1]),(floc[0]-1,floc[1]+1),(floc[0],floc[1]-1),(floc[0],floc[1]+1),(floc[0]+1,floc[1]-1),(floc[0]+1,floc[1]),(floc[0]+1,floc[1]+1)])[sqm]
        sq=iph[floc[0]-1:floc[0]+2,floc[1]-1:floc[1]+2]
        sq=sq.flatten().tolist()
        sq=np.array([*sq[:4],*sq[5:]])[sqm]
        sq=sq<sq[0]
        if np.sum(sq.astype('int'))==0:
            floc=sql[1]
        else:
            floc=sql[sq][0]
    iph[cmsk] = 0
    ig=msimg(copy.deepcopy(iph),rsig=.3,mcont=4)
    ig[cmsk] = -np.inf
    prob = 1 / (1 + np.exp(-ig[:,:,-1]))
    blabels[:,:,2] = (prob > np.random.rand(*prob.shape)).astype('int')

    ax=plt.subplot(236)
    plt.imshow(ig[:,:,0])
    ax.set_title('iph001 .08 6000')
    plt.colorbar(orientation='horizontal')

    plt.show()
