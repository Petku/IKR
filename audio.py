import matplotlib.pyplot as plt
from ikrlib import wav16khz2mfcc, logpdf_gauss, train_gauss, train_gmm, logpdf_gmm
import scipy.linalg
import numpy as np
from numpy.random import randint


train_persons = []
for number in range(1,32):
    train_person =  wav16khz2mfcc('train/' + str(number) + '/').values()
    train_persons.append(train_person)
    train_persons[number-1] = np.vstack(train_persons[number-1])

dim = train_persons[0].shape[1]

# PCA reduction to 2 dimensions

cov_tot = np.cov(np.vstack(train_persons).T, bias=True)
# take just 2 largest eigenvalues and corresponding eigenvectors
d, e = scipy.linalg.eigh(cov_tot, eigvals=(dim-2, dim-1))

train_persons_pca = []
for train_person in train_persons_pca:
    train_persons_pca.append(train_person.dot(e))

colors = ['m','k','g','c','y','r','b']
for pca, color in zip(train_persons_pca,colors):
    plt.plot(pca[:, 1], pca[:, 0], color+'.', ms=1)
    if color == 'k':
        break


plt.show()
