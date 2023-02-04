# inverse laplacian Sij=(I-A)^-1
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from segment import edge_weight
import os

def rj(xi,Sj,a1,a2,a3,a4,a5,a6,a7,a8):
    r=np.random.rand(1)
    if r<a1[xi]:
        # xj=xi+xnn[0]
        xj=(xi[0]+xnn[0][0],xi[1]+xnn[0][1])
    elif r<a2[xi]:
        xj=(xi[0]+xnn[1][0],xi[1]+xnn[1][1])
    elif r<a3[xi]:
        xj=(xi[0]+xnn[2][0],xi[1]+xnn[2][1])
    elif r<a4[xi]:
        xj=(xi[0]+xnn[3][0],xi[1]+xnn[3][1])
    elif r<a5[xi]:
        xj=(xi[0]+xnn[4][0],xi[1]+xnn[4][1])
    elif r<a6[xi]:
        xj=(xi[0]+xnn[5][0],xi[1]+xnn[5][1])
    elif r<a7[xi]:
        xj=(xi[0]+xnn[6][0],xi[1]+xnn[6][1])
    elif r<a8[xi]:
        xj=(xi[0]+xnn[7][0],xi[1]+xnn[7][1])
    else:
        print(r)
        print(a1[xi],a2[xi],a3[xi],a4[xi],a5[xi],a6[xi],a7[xi],a8[xi])

    return xj


if __name__ == "__main__":

    data_path = os.path.join(os.getcwd(), 'photos')
    im_flist = os.listdir(data_path)
    im_no = 0
    im = mpimg.imread(os.path.join(data_path, im_flist[im_no]))
    img = im[150:160, 185:200, :]

    I, J, K = img.shape
    img=img.astype('int')
    pim = np.concatenate((np.zeros((1,J,K)), img, np.zeros((1,J,K))), axis=0) #padding
    pim = np.concatenate((np.zeros((I+2,1,K)), pim,np.zeros((I+2,1,K))), axis=1)
    gamma = .9
    grad_E = pim[1:1 + I, 2:, :]-pim[1:1 + I, 1:-1, :]
    grad_S = pim[2:, 1:1 + J, :]-pim[1:-1, 1:1 + J, :]
    grad_SE = pim[2:, 2:, :] - pim[1:-1, 1:-1, :]
    grad_NE = pim[:I, 2:, :] - pim[1:-1, 1:-1,:]
    rsig = np.mean(np.abs((grad_E,grad_S,grad_SE,grad_NE)))
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
    wn=np.sum(np.dstack((w_E,w_S,w_SE,w_NE,w_W,w_N,w_NW,w_SW)),axis=2)
    w_E=w_E/wn
    w_S=w_S/wn
    w_SE =w_SE/wn
    w_NE =w_NE/wn
    w_W = w_W /wn
    w_N = w_N /wn
    w_NW =w_NW/wn
    w_SW =w_SW/wn

    a1=w_NW
    a2=a1+w_N
    a3=a2+w_NE
    a4=a3+w_E
    a5=a4+w_SE
    a6=a5+w_S
    a7=a6+w_SW
    a8=a7+w_W
    #NW,N,NE,E,SE,S,SW,W
    xnn=((-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1))
    xi=(3,3)
    Sj=np.zeros((I,J))
    for i in range(10**2):
        xj=rj(xi,Sj,a1,a2,a3,a4,a5,a6,a7,a8)
        Sj[xj]+=1
        xi=xj

    ax = plt.subplot(111)
    plt.imshow(Sj)
    ax.set_title('corr of (3,3)')
    plt.colorbar(orientation='horizontal')

    plt.show()
