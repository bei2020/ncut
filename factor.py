import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import os
from time import gmtime, strftime

if __name__ == "__main__":

    data_path = os.path.join(os.getcwd(), 'photos/gray')
    im_flist = os.listdir(data_path)
    im_no = 0
    im = mpimg.imread(os.path.join(data_path, im_flist[im_no]))
    # im = im[40:60, 10:40, :]
    im=im[40:100, 10:50]

    ip=im/1
    I, J = ip.shape
    ps=16
    m=I//ps
    n=J//ps
    ip=ip[:m*ps,:n*ps]
    ip=np.stack(np.split(ip,n,1))
    ip = np.stack(np.split(ip, m, 1))
    ip=ip-np.mean(ip,axis=(0,1))

    niter=10
    h=(np.random.rand(ps,ps)*2-1)
    ch=np.sum(h.reshape(1,1,ps,ps)*ip,(2,3))
    s2m=np.mean(ch**2)
    kT=s2m
    kT=round(kT,int(np.ceil(np.log10(1 / kT))))
    ax = plt.subplot(111)
    plt.imshow(h)
    ax.set_title('h')
    plt.colorbar(orientation='horizontal')
    plt.show()
    print('kT %f' % kT)
    def seq_change(s0,h,pats):
        for i in range(s0[0],ps,2):
            for j in range(s0[1],ps,2):
                h[i,j]=0
                ch=np.sum(h.reshape(1,1,ps,ps)*ip,(2,3))*ip[:,:,i,j]
                dq=np.mean(ch)*4
                h[i,j]=(1/(1+np.exp(-dq/kT))>np.random.rand()).astype('int')*2-1
        return h

    for _ in range(niter):
        h=seq_change((0,0),h,ip)
        h=seq_change((0,1),h,ip)
        h=seq_change((1,0),h,ip)
        h=seq_change((1,1),h,ip)



    ax = plt.subplot(141)
    plt.imshow(im)
    ax.set_title('sample')
    plt.colorbar(orientation='horizontal')
    ax = plt.subplot(142)
    plt.imshow(ch)
    ax.set_title('ch')
    plt.colorbar(orientation='horizontal')

    ax = plt.subplot(143)
    plt.imshow(h)
    ax.set_title('h')
    plt.colorbar(orientation='horizontal')

    # ax = plt.subplot(144)
    # plt.imshow(h)
    # ax.set_title('h')
    # plt.colorbar(orientation='horizontal')

    plt.show()
