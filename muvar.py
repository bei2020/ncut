import numpy as np
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt

if __name__ == "__main__":
    with open('imgs/tri_part', 'rb') as f:
        im = np.load(f)
    if len(im.shape)==3:
        I,J,K=im.shape
    else:
        I,J=im.shape
        K=1
    x = im.reshape(I*J*K, 1)
    nm = 10
    bics = np.zeros(nm)
    # bicsr = np.zeros(nm)
    for i in range(nm):
        ncomp = i + 1
        model = GaussianMixture(ncomp)
        model.fit(x)
        # Lm = np.max(np.matmul(np.multiply(
        #     np.exp(-(x - model.means_.reshape(1, ncomp)) ** 2 / (2 * model.covariances_.reshape(1, ncomp) ** 2)),
        #     1 / (np.sqrt(2 * np.pi) * model.covariances_.reshape(1, ncomp))), model.weights_))
        #bicsr[i]= -2*np.log(Lm)+ncomp*np.log(x.shape[0])
        #bics[i]= -2*np.log(Lm)+np.log(ncomp)
        bics[i] = model.bic(x)
    ncomp=np.argmin(bics)+1
    model = GaussianMixture(ncomp)
    model.fit(x)
    lb=model.predict(x).reshape(im.shape)

    ax = plt.subplot(121)
    plt.imshow(im)
    ax.set_title('img')
    plt.colorbar(orientation='horizontal')
    ax = plt.subplot(122)
    plt.imshow(lb)
    ax.set_title('class predict')
    plt.colorbar(orientation='horizontal')

    plt.show()
