import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from segment import edge_weight
import copy
import os
from time import gmtime, strftime


def msimg(img,ssig=1,rsig=None,mcont=5,init_wt=1):
    """Return mean shift image."""
    fn=strftime("%Y%b%d", gmtime())
    if init_wt==1:
        I,J,K=img.shape
        D=img.shape[2]
        gamma=.9
        alpha=.99
        lamb=1e-5
        v = np.zeros((img.shape))
        Vp = 0
        nint=2
        # 1234:ESWN, Io: DHWC, D=8 directions
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
        grad=Io-img.reshape(1,I,J,K)
        if not rsig:
            rsig=grad_m(grad)
        print('rsig %f'%rsig)
        a=rsig**2/(D+2)
        wt=edge_weight(grad,rsig*2**(1/nint))
        wt=wt/np.sum(wt)
        di=np.sum(wt,0)
        # Iok = np.sum(np.multiply(Io,wt.reshape((*wt.shape,1))),0)
        Iok = np.sum(np.multiply(Io,wt.reshape((*wt.shape,1))),0)/di.reshape((I,J,1))
        wt=edge_weight(Io-img.reshape(1,I,J,K),rsig)
        wt=wt/np.sum(wt)
        di=np.sum(wt,0)
        # Io = np.sum(np.multiply(Io,wt.reshape((*wt.shape,1))),0)
        Io = np.sum(np.multiply(Io,wt.reshape((*wt.shape,1))),0)/di.reshape((I,J,1))
        cur=Iok-Io
        fn='wt_rsig%f_%s'%(rsig,fn)
        with open('%s.npy'%fn, 'wb') as f:
            np.save(f, wt)

        # gloc=np.argmin(di[1:-1,1:-1])
        # gloc = (gloc // (J - 2) + 1, gloc % (J - 2) + 1)
        # print('gloc %d %d'%(gloc[0],gloc[1]))
        # floc=np.argmin(wt[:,gloc[0],gloc[1]])
        # floc=(gloc[0]+lij[floc][0],gloc[1]+lij[floc][1])
        # print('floc %d %d'%(floc[0],floc[1]))
        # fi = np.zeros((I,J,K))
        # fi[floc[0],floc[1],:] = 1
        # fi[gloc[0],gloc[1],:] = -1
        # img[gloc[0],gloc[1],:]=0
    else:
        fn='wt_rsig%f_%s'%(rsig,fn)
        with open('%s.npy'%fn, 'rb') as f:
            wt = np.load(f)

    def msiter(niter=1000):
        nonlocal img,v,Vp,wt,cur
        # wt[:,gloc[0],gloc[1]]=0
        # wt[:,floc[0],floc[1]]=0
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
            Io = np.sum(np.multiply(Io,wt.reshape((*wt.shape,1))),0)/di.reshape((I,J,1))
            # Io = np.sum(np.multiply(Io,wt.reshape((*wt.shape,1))),0)
            # Io+=fi
            gp=Io-(img+gamma*v)
            Vp=alpha*Vp+(1-alpha)*gp**2
            Gp=1/(lamb+np.sqrt(Vp))
            v=gamma*v+a*(np.multiply(Gp,gp)+cur)
            img+=v
            # c=np.sum(np.multiply(img,di.reshape((I,J,1))),axis=(0,1))
            # img-=c.reshape((1,1,K))/(I*J)/di.reshape((I,J,1))
            # c=np.sum(np.multiply(img**2,di.reshape((I,J,1))),axis=(0,1))
            # img/=(c**(1/2)).reshape((1,1,K))
    def msconti(ncont=0,mcont=12):
        nonlocal img
        imgc=copy.deepcopy(img)
        msiter(10)
        # if np.array_equal(np.round(np.exp(img),8),np.round(np.exp(imgc),8)):
        if np.array_equal(np.round(img,8),np.round(imgc,8)):
            print('fix found continue iter %d'%(ncont*1000))
            # print('ground ',img[gloc[0],gloc[1],:])
            return
        else:
            if ncont==mcont:
                print('no fix after continue iter %d'%(ncont*1000))
                print(np.round(img[:10,:10,0], 6))
                print(np.round(imgc[:10,:10,0], 6))
                # print('ground ',img[gloc[0],gloc[1],:])
                # print(np.round(np.exp(img[:10,:10,0]), 6))
                # print(np.round(np.exp(imgc[:10,:10,0]), 6))
                return
            print('continue')
            ncont+=1
            msiter(niter=1000)
            msconti(ncont,mcont)
    msiter()
    msconti(mcont=mcont)
    return img

def grad_m(grad):
    """Return prob mean value of gradient"""
    h, b = np.histogram(abs(grad).flatten(), bins=20)
    pk=np.argmax(h)
    nf=3
    ginf=1e-8
    vh=[None]*2
    vh[0]=np.dot(h[pk:pk+nf], b[1+pk:1+pk+nf]) / np.sum(h[pk:pk+nf])
    vh[1]=np.dot(h[pk:pk+nf+1], b[1+pk:2+pk+nf]) / np.sum(h[pk:pk+nf+1])
    for i in range(2,len(h)):
        v=np.dot(h[pk:pk+nf+i], b[1+pk:1+pk+nf+i]) / np.sum(h[pk:pk+nf+i])
        if (abs(v-vh[1])<ginf) & (abs(vh[0]+v-2*vh[1])<ginf):
            break
        vh[0]=vh[1]
        vh[1]=v
    return np.round(v,int(np.ceil(np.log10(1/v))))


if __name__=="__main__":
    # lij=((0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,-1),(-1,1))

    # with open('imgs/tri_part','rb') as f:
    #     im=np.load(f)
    # # im=im[8:18,1:11]
    # # iph=im.reshape((*im.shape,1))
    # iph=(im/np.sum(im)).reshape((*im.shape,1))
    # # iph=np.sqrt(iph).reshape((*im.shape,1))
    # # iph=np.log(iph).reshape((*im.shape,1))

    data_path=os.path.join( os.getcwd(),'photos')
    im_flist=os.listdir(data_path)
    im_no=0
    im=mpimg.imread(os.path.join(data_path,im_flist[im_no]))
    # im=im[110:150,140:190,:]
    im=im[40:60,10:50,:]
    # im=im[40:80,10:60,:]
    ime=np.einsum('ijk->k', im.astype('uint32')).reshape(1,1,im.shape[2])
    iph=im/ime
    iph[iph==0]=.0000001/np.sum(ime)
    # iph=np.sqrt(iph)

    I,J,K=iph.shape
    l=np.arange(I*J).reshape(I,J)
    ig=msimg(copy.deepcopy(iph),mcont=4)
    # ig=msimg(copy.deepcopy(iph),rsig=.1,mcont=4)
    # ig=msimg(ig,rsig=.08)

    # blabels=np.zeros((I,J,6))
    # prob = 1 / (1 + np.exp(-np.sum(np.multiply(ig,iph),axis=2)))
    # prob = 1 / (1 + np.exp(-ig[:,:,ch]))
    # blabels[:,:,0] = (prob > np.random.rand(*prob.shape)).astype('int')

    ax=plt.subplot(231)
    # plt.imshow(im[:,:,0])
    plt.imshow(im)
    ax.set_title('sample')
    plt.colorbar(orientation='horizontal')
    ax=plt.subplot(232)
    # plt.imshow(np.square(ig[:,:,0])*np.sum(im))
    plt.imshow(ig[:,:,0])
    ax.set_title('ig')
    plt.colorbar(orientation='horizontal')
    # ax=plt.subplot(233)
    # plt.imshow(blabels[:,:,0])
    # ax.set_title('seg1 4000')
    # plt.colorbar(orientation='horizontal')

    ax=plt.subplot(233)
    plt.imshow(iph[:,:,0])
    ax.set_title('iph')
    plt.colorbar(orientation='horizontal')

    # cmsk=np.sum(blabels,-1).astype('bool') # cut nodes mask
    # iph[cmsk] = -np.inf
    # iph[cmsk] = 0
    # ig=msimg(copy.deepcopy(iph),rsig=.1,mcont=4)
    # ig[cmsk] = -np.inf
    # prob = 1 / (1 + np.exp(-ig[:,:,ch]))
    # blabels[:,:,1] = (prob > np.random.rand(*prob.shape)).astype('int')
    #
    # ax=plt.subplot(234)
    # plt.imshow(ig[:,:,ch])
    # ax.set_title('ch%d iph01'%ch)
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
