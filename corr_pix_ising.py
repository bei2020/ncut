# inverse laplacian Sij=(I-A)^-1
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from segment import edge_weight
import os


if __name__ == "__main__":

    data_path = os.path.join(os.getcwd(), 'photos')
    im_flist = os.listdir(data_path)
    im_no = 0
    im = mpimg.imread(os.path.join(data_path, im_flist[im_no]))
    img = im[150:157, 188:198, :]

    I, J, K = img.shape
    img=img.astype('int')
    g_NE=np.zeros((I,J,K))
    g_E=np.zeros((I,J,K))
    g_SE=np.zeros((I,J,K))
    g_S=np.zeros((I,J,K))
    g_E[:,:-1,:] = img[:,:-1,:]-img[:, 1:, :]
    g_S[:-1,:,:] = img[:-1, :, :]-img[1:, :, :]
    g_NE[1:,:-1,:] = img[1:,:-1,:] - img[:-1, 1:,:]
    g_SE[:-1,:-1,:] = img[:-1,:-1,:] - img[1:, 1:, :]
    rsig = np.mean(np.abs((g_E,g_S,g_SE,g_NE)))
    print('rsig %f' % rsig)
    w_E = edge_weight(g_E,rsig)
    w_E[:,-1]=0
    w_S = edge_weight(g_S, rsig)
    w_S[-1,:]=0
    w_SE = edge_weight(g_SE, rsig)
    w_SE[:,-1]=0
    w_SE[-1,:]=0
    w_NE = edge_weight(g_NE, rsig)
    w_NE[:,-1]=0
    w_NE[0,:]=0
    w_W = np.hstack((np.zeros((I, 1)), w_E[:, :-1]))
    w_N = np.vstack((np.zeros((1, J)), w_S[:-1, :]))
    w_NW = np.hstack((np.zeros((I, 1)),np.vstack((np.zeros((1, J - 1)), w_SE[:-1,:-1]))))
    w_SW = np.hstack((np.zeros((I, 1)), np.vstack((w_NE[1:,:-1], np.zeros((1, J - 1))))))
    wn=np.sum(np.dstack((w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW)),axis=2)
    w_E=w_E/wn
    w_S=w_S/wn
    w_SE =w_SE/wn
    w_NE =w_NE/wn
    w_W = w_W /wn
    w_N = w_N /wn
    w_NW =w_NW/wn
    w_SW =w_SW/wn

    beta=1

    xt=(3,3)
    m=100
    Sij=np.zeros((I,J))
    Si=np.random.randint(-1,1,size=(I,J))
    Si[Si==0]=1
    pSi=np.zeros((I+2,J+2))
    pSi[1:-1,1:-1]=Si
    Ebt=0
    # Em=[None]*m
    for d in range(m):
        for i in range(I):
            for j in range(J):
                hi=w_E[i,j]*pSi[i+1,j+2]+ w_S[i,j]*pSi[i+2,j+1]+w_W[i,j]*pSi[i+1,j]+w_N[i,j]*pSi[i,j+1]+\
                   w_SE[i,j]*pSi[i+2,j+2]+w_SW[i,j]*pSi[i+2,j]+w_NW[i,j]*pSi[i,j]+w_NE[i,j]*pSi[i,j+2]
                pSi[1+i,1+j]=((1/(1+np.exp(-beta*hi)))>np.random.rand(1)).astype('int')*2-1
        Si=pSi[1:-1,1:-1]
        sx=(Si[xt[0],xt[1]]==Si).astype('int')
        Sij+=sx
        E=np.zeros((I,J))
        E[1:,:-1] =w_NE[1:,:-1]*Si[:-1,1:]
        E[:,:-1]+=w_E[:,:-1]*Si[:,1:]
        E[:-1,:-1]+=w_SE[:-1,:-1]*Si[1:,1:]
        E[:-1,:] +=w_S[:-1]*Si[1:,:]
        E=np.sum(Si*E)/(I*J)
        # Em[d]=E
        Ebt+=E

    Sij/=m
    Ebt/=m


    ax = plt.subplot(121)
    plt.imshow(img)
    ax.set_title('img')
    plt.colorbar(orientation='horizontal')
    ax = plt.subplot(122)
    plt.imshow(Sij)
    ax.set_title('corr of %s'%('xt'))
    plt.colorbar(orientation='horizontal')

    plt.show()
