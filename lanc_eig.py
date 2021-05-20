# coding: utf-8
# w,v=eigs(L,k=2,M=D,which='SM')
# w,v=eigs(L,k=2,M=D,which='SM')
# L
# w
# v[:,0]
# v[:,1]

import numpy as np
import copy

def lanc_pair(w,Ls,km=4):
    v=np.matmul(Ls,w)
    S=[None]*km
    S[0]=copy.deepcopy(v)#S[0]=v/1
    alpha=np.dot(w,v)
    v-=alpha*w
    # beta=np.dot(v,v)
    beta=np.sqrt(np.dot(v,v))
    W=[None]*km
    V=[None]*km
    alp=[None]*km
    W[0]=w
    V[0]=copy.deepcopy(v)
    alp[0]=alpha
    k=1
    while beta!=0 and k<km:
        t=w
        w=v/beta
        W[k]=w
        v=np.matmul(Ls,w)
        S[k]=copy.deepcopy(v)
        v-=beta*t
        alpha=np.dot(w,v)
        v-=alpha*w
        beta=np.sqrt(np.dot(v,v))
        V[k]=copy.deepcopy(v)
        alp[k]=alpha
        k+=1
    return np.array(W).T,np.array(V),np.array(S)

def givens(a,b):
    if b==0:
        c=1
        s=0
    else:
        if abs(b)>abs(a):
            tau=-a/b
            s=1/np.sqrt(1+tau**2)
            c=s*tau
        else:
            tau=-b/a
            c=1/np.sqrt(1+tau**2)
            s=c*tau
    return c,s

def SQR(T,tol):
    # 8.3.2 Implicit Symmetric QR Step with Wilkinson Shift
    d=(T[-2,-2]-T[-1,-1])/2
    miu=T[-1,-1]-T[-1,-2]**2/(d+np.sign(d)*np.sqrt(d**2+T[-1,-2]**2))
    x=T[0,0]-miu
    z=T[1,0]
    Z=np.identity(T.shape[0])
    for k in range(T.shape[0]-1):
        c,s=givens(x,z)
        G=np.array([[c,s],[-s,c]])
        m=min(k+3,T.shape[1])
        p=max(0,k-1)
        T[k:k+2,p:m]=np.matmul(G.T,T[k:k+2,p:m])
        m=min(k+3,T.shape[0])
        T[p:m,k:k+2]=np.matmul(T[p:m,k:k+2],G)
        if k<T.shape[1]-2:
            x=T[k+1,k]
            z=T[k+2,k]
            # print(k,x,z)
        Gn=np.identity(T.shape[0])
        Gn[k:k+2,k:k+2]=G
        Z=np.matmul(Z,Gn)
    return T,Z

def eiges(Ls,k=4):
    w=np.zeros(Ls.shape[1])
    w[0]=1
    W,V,S=lanc_pair(w,Ls,k)
    T=np.matmul(W.T,S.T) #T=np.matmul(W.T,np.matmul(Ls,W))

    tol=1e-16
    n=T.shape[0]
    miter=2000
    print('total iter%d'%miter)
    q=0
    while (q<n-1) & (miter>0):
        for i in range(n-1):
            if abs(T[i,i+1])<=tol*(abs(T[i,i])+abs(T[i+1,i+1])):
                T[i,i+1]=0
                T[i+1,i]=0
        for p in range(n):
            if T[p,p+1]!=0:
                break
        for q in range(n):
            if T[n-1-q,n-2-q]!=0:
                break
        # print('p:%d q:%d'%(p,q))
        if q<n-1:
            T[p:n-q,p:n-q],Z=SQR(T[p:n-q,p:n-q],tol)
            W[:,p:n-q]=np.matmul(W[:,p:n-q],Z)
        miter-=1
    print('count down:%d'%miter)
    return np.diag(T),W

if __name__=='__main__':
    A=np.array([[1,1,0,0,0],[1,0,0,1,0],[0,1,1,0,0],[0,0,1,1,0],[0,0,1,0,1]])
    di=np.diag_indices(5)
    A[di]=0
    A[[1,3,2],[2,1,3]]=1
    A[2,4]=1
    A.T==A
    A=np.vstack((A,np.zeros((1,5))))
    A= np.hstack((A,np.zeros((6,1))))
    A[[1,5],[5,1]]=1
    D=np.diag(np.sum(A,0))
    L=D-A
    d = np.diag(D)
    Ls=np.multiply(np.multiply((d**(-1/2)).reshape(6, 1), L), (d**(-1/2)).reshape(1, 6))

    k=4
    # w=np.zeros(Ls.shape[1])
    # w[0]=1
    # W,V,S=lanc_pair(w,Ls,k)
    # T=np.matmul(W.T,S.T) #T=np.matmul(W.T,np.matmul(Ls,W))

    w,v=eiges(Ls,k)

