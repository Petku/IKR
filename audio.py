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


plt.plot(train_persons[0][:, 1], train_persons[0][:, 0], 'b.', ms=1)
plt.plot(train_persons[1][:, 1], train_persons[1][:, 0], 'r.', ms=1)
plt.plot(train_persons[2][:, 1], train_persons[2][:, 0], 'y.', ms=1)
plt.plot(train_persons[3][:, 1], train_persons[3][:, 0], 'c.', ms=1)
plt.plot(train_persons[4][:, 1], train_persons[4][:, 0], 'g.', ms=1)
plt.plot(train_persons[5][:, 1], train_persons[5][:, 0], 'k.', ms=1)
plt.plot(train_persons[6][:, 1], train_persons[6][:, 0], 'm.', ms=1)
plt.show()


lens_of_persons = []
for person in train_persons:
    lens_of_persons.append(len(person))


covariance_weights = []
for length, training_data in zip(lens_of_persons,train_persons):
    tmp = length * np.cov(training_data.T, bias=True)
    covariance_weights.append(tmp)

sum_cov = 0
for cov_w in covariance_weights:
    sum_cov += cov_w

cov_wc = sum_cov / np.sum(lens_of_persons)
cov_ac = cov_tot - cov_wc
d, e = scipy.linalg.eigh(cov_ac, cov_wc, eigvals=(dim-1, dim-1))
plt.figure()
junk = plt.hist(train_persons[0].dot(e), 40, histtype='step', color='b', normed=True)
junk = plt.hist(train_persons[1].dot(e), 40, histtype='step', color='r', normed=True)
plt.show()
