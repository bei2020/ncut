import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from segment import edge_weight,grad_m
from util import mpb_maxloc
from time import gmtime, strftime

def SE_it(gc,img,w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW,di):
    I, J= img.shape
    k=3
    miter=10
    ni=(I-gc[0]-1)//k
    nj=(J-gc[1]-1)//k
    m=min(ni,nj)
    if m==0:
        return img
    for j in range(1,nj):
        for _ in range(miter):
            img[gc[0]:gc[0]+k,gc[1]+j*k:gc[1]+k+j*k]=((np.multiply(w_W[gc[0]:gc[0]+k,gc[1]+j*k:gc[1]+j*k+k],img[gc[0]:gc[0]+k,gc[1]+j*k-1:gc[1]+j*k+k-1])+np.multiply(w_N[gc[0]:gc[0]+k,gc[1]+j*k:gc[1]+j*k+k],img[gc[0]-1:gc[0]+k-1,gc[1]+j*k:gc[1]+j*k+k])+np.multiply(w_E[gc[0]:gc[0]+k,gc[1]+j*k:gc[1]+j*k+k],img[gc[0]:gc[0]+k,gc[1]+j*k+1:gc[1]+j*k+k+1])+np.multiply(w_S[gc[0]:gc[0]+k,gc[1]+j*k:gc[1]+j*k+k],img[gc[0]+1:gc[0]+k+1,gc[1]+j*k:gc[1]+j*k+k])+np.multiply(w_SE[gc[0]:gc[0]+k,gc[1]+j*k:gc[1]+j*k+k],img[gc[0]+1:gc[0]+k+1,gc[1]+j*k+1:gc[1]+j*k+k+1])+np.multiply(w_SW[gc[0]:gc[0]+k,gc[1]+j*k:gc[1]+j*k+k],img[gc[0]+1:gc[0]+k+1,gc[1]+j*k-1:gc[1]+j*k+k-1])+np.multiply(w_NW[gc[0]:gc[0]+k,gc[1]+j*k:gc[1]+j*k+k],img[gc[0]-1:gc[0]+k-1,gc[1]+j*k-1:gc[1]+j*k+k-1])+np.multiply(w_NE[gc[0]:gc[0]+k,gc[1]+j*k:gc[1]+j*k+k],img[gc[0]-1:gc[0]+k-1,gc[1]+j*k+1:gc[1]+j*k+k+1]))*2-di[gc[0]:gc[0]+k,gc[1]+j*k:gc[1]+k+j*k])>0
    for i in range(1,ni):
        for _ in range(miter):
            img[gc[0]+i*k:gc[0]+i*k+k,gc[1]:gc[1]+k]=((np.multiply(w_W[gc[0]+i*k:gc[0]+i*k+k,gc[1]:gc[1]+k],img[gc[0]+i*k:gc[0]+i*k+k,gc[1]-1:gc[1]+k-1])+np.multiply(w_N[gc[0]+i*k:gc[0]+i*k+k,gc[1]:gc[1]+k],img[gc[0]+i*k-1:gc[0]+i*k+k-1,gc[1]:gc[1]+k])+np.multiply(w_E[gc[0]+i*k:gc[0]+i*k+k,gc[1]:gc[1]+k],img[gc[0]+i*k:gc[0]+i*k+k,gc[1]+1:gc[1]+k+1])+np.multiply(w_S[gc[0]+i*k:gc[0]+i*k+k,gc[1]:gc[1]+k],img[gc[0]+i*k+1:gc[0]+i*k+k+1,gc[1]:gc[1]+k])+np.multiply(w_SE[gc[0]+i*k:gc[0]+i*k+k,gc[1]:gc[1]+k],img[gc[0]+i*k+1:gc[0]+i*k+k+1,gc[1]+1:gc[1]+k+1])+np.multiply(w_SW[gc[0]+i*k:gc[0]+i*k+k,gc[1]:gc[1]+k],img[gc[0]+i*k+1:gc[0]+i*k+k+1,gc[1]-1:gc[1]+k-1])+np.multiply(w_NW[gc[0]+i*k:gc[0]+i*k+k,gc[1]:gc[1]+k],img[gc[0]+i*k-1:gc[0]+i*k+k-1,gc[1]-1:gc[1]+k-1])+np.multiply(w_NE[gc[0]+i*k:gc[0]+i*k+k,gc[1]:gc[1]+k],img[gc[0]+i*k-1:gc[0]+i*k+k-1,gc[1]+1:gc[1]+k+1]))*2-di[gc[0]+i*k:gc[0]+i*k+k,gc[1]:gc[1]+k])>0
    for i in range(0,m-1):
        for r in range(i+1,ni):
            for _ in range(miter):
                img[gc[0]+k*r:gc[0]+k*r+k,gc[1]+(i+1)*k:gc[1]+(i+1)*k+k]=((np.multiply(w_W[gc[0]+k*r:gc[0]+k*r+k,gc[1]+(i+1)*k:gc[1]+(i+1)*k+k],img[gc[0]+k*r:gc[0]+k*r+k,gc[1]+(i+1)*k-1:gc[1]+(i+1)*k+k-1])+np.multiply(w_N[gc[0]+k*r:gc[0]+k*r+k,gc[1]+(i+1)*k:gc[1]+(i+1)*k+k],img[gc[0]+k*r-1:gc[0]+k*r+k-1,gc[1]+(i+1)*k:gc[1]+(i+1)*k+k])+np.multiply(w_E[gc[0]+k*r:gc[0]+k*r+k,gc[1]+(i+1)*k:gc[1]+(i+1)*k+k],img[gc[0]+k*r:gc[0]+k*r+k,gc[1]+(i+1)*k+1:gc[1]+(i+1)*k+k+1])+np.multiply(w_S[gc[0]+k*r:gc[0]+k*r+k,gc[1]+(i+1)*k:gc[1]+(i+1)*k+k],img[gc[0]+k*r+1:gc[0]+k*r+k+1,gc[1]+(i+1)*k:gc[1]+(i+1)*k+k])+np.multiply(w_SE[gc[0]+k*r:gc[0]+k*r+k,gc[1]+(i+1)*k:gc[1]+(i+1)*k+k],img[gc[0]+k*r+1:gc[0]+k*r+k+1,gc[1]+(i+1)*k+1:gc[1]+(i+1)*k+k+1])+np.multiply(w_SW[gc[0]+k*r:gc[0]+k*r+k,gc[1]+(i+1)*k:gc[1]+(i+1)*k+k],img[gc[0]+k*r+1:gc[0]+k*r+k+1,gc[1]+(i+1)*k-1:gc[1]+(i+1)*k+k-1])+np.multiply(w_NW[gc[0]+k*r:gc[0]+k*r+k,gc[1]+(i+1)*k:gc[1]+(i+1)*k+k],img[gc[0]+k*r-1:gc[0]+k*r+k-1,gc[1]+(i+1)*k-1:gc[1]+(i+1)*k+k-1])+np.multiply(w_NE[gc[0]+k*r:gc[0]+k*r+k,gc[1]+(i+1)*k:gc[1]+(i+1)*k+k],img[gc[0]+k*r-1:gc[0]+k*r+k-1,gc[1]+(i+1)*k+1:gc[1]+(i+1)*k+k+1]))*2-di[gc[0]+k*r:gc[0]+k*r+k,gc[1]+(i+1)*k:gc[1]+(i+1)*k+k])>0
        for r in range(i+1,nj):
            for _ in range(miter):
                img[gc[0]+(i+1)*k:gc[0]+(i+1)*k+k,gc[1]+k*r:gc[1]+k*r+k]=((np.multiply(w_W[gc[0]+(i+1)*k:gc[0]+(i+1)*k+k,gc[1]+k*r:gc[1]+k*r+k],img[gc[0]+(i+1)*k:gc[0]+(i+1)*k+k,gc[1]+k*r-1:gc[1]+k*r+k-1])+np.multiply(w_N[gc[0]+(i+1)*k:gc[0]+(i+1)*k+k,gc[1]+k*r:gc[1]+k*r+k],img[gc[0]+(i+1)*k-1:gc[0]+(i+1)*k+k-1,gc[1]+k*r:gc[1]+k*r+k])+np.multiply(w_E[gc[0]+(i+1)*k:gc[0]+(i+1)*k+k,gc[1]+k*r:gc[1]+k*r+k],img[gc[0]+(i+1)*k:gc[0]+(i+1)*k+k,gc[1]+k*r+1:gc[1]+k*r+k+1])+np.multiply(w_S[gc[0]+(i+1)*k:gc[0]+(i+1)*k+k,gc[1]+k*r:gc[1]+k*r+k],img[gc[0]+(i+1)*k+1:gc[0]+(i+1)*k+k+1,gc[1]+k*r:gc[1]+k*r+k])+np.multiply(w_SE[gc[0]+(i+1)*k:gc[0]+(i+1)*k+k,gc[1]+k*r:gc[1]+k*r+k],img[gc[0]+(i+1)*k+1:gc[0]+(i+1)*k+k+1,gc[1]+k*r+1:gc[1]+k*r+k+1])+np.multiply(w_SW[gc[0]+(i+1)*k:gc[0]+(i+1)*k+k,gc[1]+k*r:gc[1]+k*r+k],img[gc[0]+(i+1)*k+1:gc[0]+(i+1)*k+k+1,gc[1]+k*r-1:gc[1]+k*r+k-1])+np.multiply(w_NW[gc[0]+(i+1)*k:gc[0]+(i+1)*k+k,gc[1]+k*r:gc[1]+k*r+k],img[gc[0]+(i+1)*k-1:gc[0]+(i+1)*k+k-1,gc[1]+k*r-1:gc[1]+k*r+k-1])+np.multiply(w_NE[gc[0]+(i+1)*k:gc[0]+(i+1)*k+k,gc[1]+k*r:gc[1]+k*r+k],img[gc[0]+(i+1)*k-1:gc[0]+(i+1)*k+k-1,gc[1]+k*r+1:gc[1]+k*r+k+1]))*2-di[gc[0]+(i+1)*k:gc[0]+(i+1)*k+k,gc[1]+k*r:gc[1]+k*r+k])>0

    return img

def msimg(img, ssig=1, rsig=None, mcont=5, init_wt=1):
    """Return mean shift image."""
    fn = strftime("%Y%b%d", gmtime())
    I, J, K = img.shape
    img=img.astype('int')
    pim = np.concatenate((np.zeros((1,J,K)), img, np.zeros((1,J,K))), axis=0) #padding
    pim = np.concatenate((np.zeros((I+2,1,K)), pim,np.zeros((I+2,1,K))), axis=1)
    if init_wt == 1:
        # 1234:ESWN, Io: DHWC, D=8 directions
        grad_E = -pim[1:1 + I, 1:-1, :] + pim[1:1 + I, 2:, :]
        grad_S = -pim[1:-1, 1:1 + J, :] + pim[2:, 1:1 + J, :]
        grad_SE = pim[2:, 2:, :] - pim[1:-1, 1:-1, :]
        grad_NE = pim[:I, 2:, :] - pim[1:-1, 1:-1,:]
        if not rsig:
            rsig = grad_m(np.hstack((grad_E,grad_S,grad_SE,grad_NE)))
        print('rsig %f' % rsig)
        w_E = edge_weight(grad_E,rsig)
        w_E[:,-1]=0
        w_S = edge_weight(grad_S, rsig)
        w_S[-1,:]=0
        w_SE = edge_weight(grad_SE, rsig)
        w_SE[:,-1]=0
        w_SE[-1,:]=0
        w_NE = edge_weight(grad_NE, rsig)
        w_NE[:,-1]=0
        w_NE[0,:]=0
        w_W = np.hstack((np.zeros((I, 1)), w_E[:, :-1]))
        w_N = np.vstack((np.zeros((1, J)), w_S[:-1, :]))
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
        fn = 'wt_rsig%f_%s' % (rsig, fn)
        with open('%s.npy' % fn, 'wb') as f:
            np.save(f, np.stack((w_E,w_S,w_SE,w_NE)))

    else:
        fn = 'wt_rsig%f_%s' % (rsig, fn)
        with open('%s.npy' % fn, 'rb') as f:
            wt = np.load(f)
        w_E,w_S,w_SE,w_NE=wt
        w_W = np.hstack((np.zeros((I, 1)), w_E[:, :-1]))
        w_N = np.vstack((np.zeros((1, J)), w_S[:-1, :,]))
        w_NW = np.hstack((np.zeros((I, 1)),np.vstack((np.zeros((1, J - 1)), w_SE[:-1,:-1]))))
        w_SW = np.hstack((np.zeros((I, 1)), np.vstack((w_NE[1:,:-1], np.zeros((1, J - 1))))))
        di=np.sum(np.stack((w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW),0),0)

    gloc=mpb_maxloc(img)
    print('gloc %d %d'%(gloc[0],gloc[1]))
    floc=np.argmin(np.stack((w_E,w_S,w_W,w_N,w_SE,w_SW,w_NW,w_NE))[:,gloc[0],gloc[1]])
    floc=(gloc[0]+lij[floc][0],gloc[1]+lij[floc][1])
    print('floc %d %d'%(floc[0],floc[1]))

    bim = np.ones((I,J)).astype('int')
    bim[gloc[0],gloc[1]] = 0
    bim[np.sum(abs(img-img[gloc]),-1)<np.sum(abs(img-img[floc]),-1)]=0 #gloc=0

    gc=((gloc[0],floc[0])[gloc[0]>floc[0]],(gloc[1],floc[1])[gloc[1]>floc[1]])
    print('top left loc of gf %d %d'%(gc[0],gc[1]))
    k=3
    #source
    for _ in range(10):
        bim[gc[0]:gc[0]+k,gc[1]:gc[1]+k]=((np.multiply(w_W[gc[0]:gc[0]+k,gc[1]:gc[1]+k],bim[gc[0]:gc[0]+k,gc[1]-1:gc[1]+k-1])+np.multiply(w_N[gc[0]:gc[0]+k,gc[1]:gc[1]+k],bim[gc[0]-1:gc[0]+k-1,gc[1]:gc[1]+k])+np.multiply(w_E[gc[0]:gc[0]+k,gc[1]:gc[1]+k],bim[gc[0]:gc[0]+k,gc[1]+1:gc[1]+k+1])+np.multiply(w_S[gc[0]:gc[0]+k,gc[1]:gc[1]+k],bim[gc[0]+1:gc[0]+k+1,gc[1]:gc[1]+k])+np.multiply(w_SE[gc[0]:gc[0]+k,gc[1]:gc[1]+k],bim[gc[0]+1:gc[0]+k+1,gc[1]+1:gc[1]+k+1])+np.multiply(w_SW[gc[0]:gc[0]+k,gc[1]:gc[1]+k],bim[gc[0]+1:gc[0]+k+1,gc[1]-1:gc[1]+k-1])+np.multiply(w_NW[gc[0]:gc[0]+k,gc[1]:gc[1]+k],bim[gc[0]-1:gc[0]+k-1,gc[1]-1:gc[1]+k-1])+np.multiply(w_NE[gc[0]:gc[0]+k,gc[1]:gc[1]+k],bim[gc[0]-1:gc[0]+k-1,gc[1]+1:gc[1]+k+1]))*2-di[gc[0]:gc[0]+k,gc[1]:gc[1]+k])>0
    bim=SE_it(gc,bim,w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW,di)
    # NW
    bim=np.rot90(bim,2)
    rg=(gc[0]+k-1,gc[1]+k-1)
    rg=(I-rg[0]-1,J-rg[1]-1)
    rw_E=np.rot90(w_W ,2)
    rw_S=np.rot90(w_N ,2)
    rw_SE=np.rot90(w_NW,2)
    rw_NE=np.rot90(w_SW,2)
    rw_N=np.rot90(w_S ,2)
    rw_W=np.rot90(w_E ,2)
    rw_NW=np.rot90(w_SE,2)
    rw_SW=np.rot90(w_NE,2)
    rdi=np.rot90(di ,2)
    bim=SE_it(rg,bim,rw_E,rw_S,rw_SE,rw_NE,rw_W,rw_N,rw_NW,rw_SW,rdi)
    bim=np.rot90(bim,2)
    # NE
    bim=np.rot90(bim,1)
    rg=(gc[0]+k-1,gc[1])
    rg=(J-rg[1]-1,rg[0])
    rw_E=np.rot90(w_S ,1)
    rw_S=np.rot90(w_W ,1)
    rw_SE=np.rot90(w_SW,1)
    rw_NE=np.rot90(w_SE,1)
    rw_N=np.rot90(w_E ,1)
    rw_W=np.rot90(w_N ,1)
    rw_NW=np.rot90(w_NE,1)
    rw_SW=np.rot90(w_NW,1)
    rdi=np.rot90(di ,1)
    bim=SE_it(rg,bim,rw_E,rw_S,rw_SE,rw_NE,rw_W,rw_N,rw_NW,rw_SW,rdi)
    bim=np.rot90(bim,-1)
    # SW
    bim=np.rot90(bim,-1)
    rg=(gc[0],gc[1]+k-1)
    rg=(rg[1],I-1-rg[0])
    rw_E=np.rot90(w_N ,-1)
    rw_S=np.rot90(w_E ,-1)
    rw_SE=np.rot90(w_NE,-1)
    rw_NE=np.rot90(w_NW,-1)
    rw_W=np.rot90(w_S ,-1)
    rw_N=np.rot90(w_W ,-1)
    rw_NW=np.rot90(w_SW,-1)
    rw_SW=np.rot90(w_SE,-1)
    rdi=np.rot90(di ,-1)
    bim=SE_it(rg,bim,rw_E,rw_S,rw_SE,rw_NE,rw_W,rw_N,rw_NW,rw_SW,rdi)
    bim=np.rot90(bim,1)

    return bim


if __name__ == "__main__":
    #E,S,W,N,SE,SW,NW,NE
    lij=((0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,-1),(-1,1))

    data_path = os.path.join(os.getcwd(), 'photos')
    im_flist = os.listdir(data_path)
    # im_no = 0
    im_no = 4
    im = mpimg.imread(os.path.join(data_path, im_flist[im_no]))
    # im=im[:220,120:380,:]
    # im=im[110:150,140:190,:]
    # im= im[400:410, 610:625, :]
    # im= im[400:440, 610:650, :]
    # im=im[40:80,10:60,:]
    im = im[40:60, 10:40, :]

    ig = msimg(im/1, mcont=0)

    ax = plt.subplot(121)
    # plt.imshow(im[:,:,0])
    plt.imshow(im)
    ax.set_title('sample')
    plt.colorbar(orientation='horizontal')
    ax = plt.subplot(122)
    plt.imshow(ig)
    ax.set_title('one part')
    plt.colorbar(orientation='horizontal')

    plt.show()
