# inverse laplacian Sij=(I-A)^-1
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from segment import edge_weight
import os

def rj(xi,a1,a2,a3,a4,a5,a6,a7,a8):
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

    #NW,N,NE,E,SE,S,SW,W
    xnn=((-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1))
    beta=100
    a1=w_NW
    a2=a1+w_N
    a3=a2+w_NE
    a4=a3+w_E
    a5=a4+w_SE
    a6=a5+w_S
    a7=a6+w_SW
    a8=a7+w_W


    m=100
    xi=(3,3)
    Sij=np.zeros((I,J,I,J))
    S2=np.zeros((I,J,I,J))
    xj=rj(xi,a1,a2,a3,a4,a5,a6,a7,a8)
    Sij[xi[0],xi[1],xj[0],xj[1]]+=1
    xi_1=xi
    xi=xj
    for _ in range(m):
        xj=rj(xi,a1,a2,a3,a4,a5,a6,a7,a8)
        Sij[xi[0],xi[1],xj[0],xj[1]]+=1
        S2[xi_1[0],xi_1[1],xj[0],xj[1]]+=1
        xi_1=xi
        xi=xj
    print(np.sum(Sij))
    print(np.sum(S2))
    ni=np.sum(Sij,axis=(2,3))
    gij=(Sij+S2)/ni.reshape((I,J,1,1))


    # # nr=10*4
    # nr=10
    # sd=20
    # x=[None]*sd
    # Np=np.zeros((I,J))
    # for _ in range(nr):
    #     x[0]=(3,3)
    #     for s in range(sd-1):
    #         x[s+1]=rj(x[s],a1,a2,a3,a4,a5,a6,a7,a8)
    #     for i in range(sd-1):
    #     Np[np.array(x[1:-1])[:,0],np.array(x[1:-1])[:,1]]+=1
    # Sij/=(nr+Np).reshape(I,J,1,1)

    ax = plt.subplot(121)
    plt.imshow(img)
    ax.set_title('img')
    plt.colorbar(orientation='horizontal')
    ax = plt.subplot(122)
    plt.imshow(Sij[xi[0],xi[1]])
    # plt.imshow(Sj)
    ax.set_title('corr of %d,%d'%(xi))
    plt.colorbar(orientation='horizontal')

    plt.show()
