import matplotlib.pyplot as plt
from ikrlib import wav16khz2mfcc, logpdf_gauss, train_gauss, train_gmm, logpdf_gmm
import scipy.linalg
import numpy as np
from numpy.random import randint

count_of_gausses_for_one_person = 13
count_of_people_to_distinct = 31

train_persons = []
for number in range(1, 32):
    train_person = wav16khz2mfcc('train/' + str(number) + '/').values()
    train_persons.append(train_person)
    train_persons[number-1] = np.vstack(train_persons[number-1])

tests_persons = []
for number in range(1, 32):
    test_person = wav16khz2mfcc('dev/' + str(number) + '/')
    tests_persons.append(test_person)

dim = train_persons[0].shape[1]

# PCA reduction to 2 dimensions

# cov_tot = np.cov(np.vstack(train_persons).T, bias=True)
# # take just 2 largest eigenvalues and corresponding eigenvectors
# d, e = scipy.linalg.eigh(cov_tot, eigvals=(dim-2, dim-1))
#
# train_persons_pca = []
# for train_person in train_persons_pca:
#     train_persons_pca.append(train_person.dot(e))
#
#
# plt.plot(train_persons[0][:, 1], train_persons[0][:, 0], 'b.', ms=1)
# plt.plot(train_persons[1][:, 1], train_persons[1][:, 0], 'r.', ms=1)
# plt.plot(train_persons[2][:, 1], train_persons[2][:, 0], 'y.', ms=1)
# plt.plot(train_persons[3][:, 1], train_persons[3][:, 0], 'c.', ms=1)
# plt.plot(train_persons[4][:, 1], train_persons[4][:, 0], 'g.', ms=1)
# plt.plot(train_persons[5][:, 1], train_persons[5][:, 0], 'k.', ms=1)
# plt.plot(train_persons[6][:, 1], train_persons[6][:, 0], 'm.', ms=1)
# plt.show()
#
#
# lens_of_persons = []
# for person in train_persons:
#     lens_of_persons.append(len(person))
#
#
# covariance_weights = []
# for length, training_data in zip(lens_of_persons,train_persons):
#     tmp = length * np.cov(training_data.T, bias=True)
#     covariance_weights.append(tmp)
#
# sum_cov = 0
# for cov_w in covariance_weights:
#     sum_cov += cov_w
#
# cov_wc = sum_cov / np.sum(lens_of_persons)
# cov_ac = cov_tot - cov_wc
# d, e = scipy.linalg.eigh(cov_ac, cov_wc, eigvals=(dim-1, dim-1))
# plt.figure()
# junk = plt.hist(train_persons[0].dot(e), 40, histtype='step', color='b', normed=True)
# junk = plt.hist(train_persons[1].dot(e), 40, histtype='step', color='r', normed=True)
# plt.show()
#
# apriory_prob_arr = np.full((1, 31), 0.03)


class PersonGmm(object):
    __slots__ = ['mean', 'cov', 'weight']

    def __init__(self, mean, cov, weight):
        self.cov = cov
        self.mean = mean
        self.weight = weight


persons_gmms = []

for person in train_persons:
    mean_values_of_gaussians = person[randint(1, len(person), count_of_gausses_for_one_person)]
    covariance_matrices_person = [np.var(person, axis=0)] * count_of_gausses_for_one_person
    weight_of_gauss_in_gmm = np.ones(count_of_gausses_for_one_person)/count_of_gausses_for_one_person

    persons_gmms.append(PersonGmm(mean_values_of_gaussians, covariance_matrices_person, weight_of_gauss_in_gmm))

#training
for i in range(30):

    for j in range(count_of_people_to_distinct):
        W, M, C, TTL = train_gmm(train_persons[j], persons_gmms[j].weight, persons_gmms[j].mean, persons_gmms[j].cov)
        persons_gmms[j].weight = W
        persons_gmms[j].cov = C
        persons_gmms[j].mean = M

        print("Iteration: {0}, TTL: {1}, Person{2}.".format(i, TTL, j))


score = []
ll_values = []
for i in range(31):
    for name, tst in tests_persons[i].items():
        del score[:]
        del ll_values[:]
        score.append(name)
        for gmm in persons_gmms:
            llv = logpdf_gmm(tst, gmm.weight, gmm.mean, gmm.cov)
            ll_values.append(sum(llv) + np.log(0.03))
        score.append(np.argmax(ll_values) + 1)
        score.extend(ll_values)
        print score

