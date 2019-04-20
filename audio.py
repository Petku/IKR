from ikrlib import wav16khz2mfcc, train_gmm, logpdf_gmm
import numpy as np
from numpy.random import randint

count_of_gausses_for_one_person = 8
count_of_people_to_distinct = 31

train_persons = []
for number in range(1, 32):
    train_person = wav16khz2mfcc('train/' + str(number) + '/').values()
    train_persons.append(train_person)
    train_persons[number-1] = np.vstack(train_persons[number-1])

test_persons = wav16khz2mfcc('eval/')
dim = train_persons[0].shape[1]


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


def get_name(name):
    p = name.split('/')
    return p[1].split(".wav")[0]


score = []
ll_values = []
hit_ratio = 0
alltest = 0

with open("audio_GMM", "w") as f:
    for name, tst in test_persons.items():
        del score[:]
        del ll_values[:]
        name = get_name(name)
        score.append(name)
        for gmm in persons_gmms:
            llv = logpdf_gmm(tst, gmm.weight, gmm.mean, gmm.cov)
            ll_values.append(sum(llv) + np.log(0.03))
        score.append(np.argmax(ll_values) + 1)

        score.extend(ll_values)
        final_string = " ".join([str(data) for data in score])
        f.write(final_string + '\n')
        alltest += 1

