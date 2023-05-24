# initial label diffuse
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

    p_NW=np.zeros((I,J))
    p_N=np.zeros((I,J))
    p_NE=np.zeros((I,J))
    p_E=np.zeros((I,J))
    p_SE=np.zeros((I,J))
    p_S=np.zeros((I,J))
    p_SW=np.zeros((I,J))
    p_W=np.zeros((I,J))
    Si=np.random.randint(-1,1,size=(I,J))
    Si[Si==0]=1
    t=100
    for _ in range(t):
        r=np.random.rand(I,J)
        p_NW=(w_NW>r).astype('int')
        p_N =(w_N >r).astype('int')
        p_NE=(w_NE>r).astype('int')
        p_E =(w_E >r).astype('int')
        p_SE=(w_SE>r).astype('int')
        p_S =(w_S >r).astype('int')
        p_SW=(w_SW>r).astype('int')
        p_W =(w_W >r).astype('int')
        m_W =p_W[:,1:]*Si[:,1:]
        m_SW=p_SW[:-1,1:]*Si[:-1,1:]
        m_N =p_N[1:,:]*Si[1:,:]
        m_NW=p_NW[1:,1:]*Si[1:,1:]
        m_S =p_S[:-1,:]*Si[:-1,:]
        m_E =p_E[:,:-1]*Si[:,:-1]
        m_SE=p_SE[:-1,:-1]*Si[:-1,:-1]
        m_NE=p_NE[1:,:-1]*Si[1:,:-1]
        Si[:,:-1]=   m_W
        Si[1:,:-1]+= m_SW
        Si[:-1,:]+=  m_N
        Si[:-1,:-1]+=m_NW
        Si[1:,:]+=   m_S
        Si[:,1:]+=   m_E
        Si[1:,1:]+=  m_SE
        Si[:-1,1:]+= m_NE
        Si[Si>0]=1
        Si[Si<0]=-1

    ax = plt.subplot(121)
    plt.imshow(img)
    ax.set_title('img')
    plt.colorbar(orientation='horizontal')
    ax = plt.subplot(122)
    plt.imshow(Si)
    ax.set_title('Si')
    plt.colorbar(orientation='horizontal')

    plt.show()
