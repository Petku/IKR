import matplotlib.pyplot as plt
from ikrlib import wav16khz2mfcc, logpdf_gauss, train_gauss, train_gmm, logpdf_gmm
import scipy.linalg
import numpy as np
from numpy.random import randint


def load_data_with_name(path_prefix="train/", data_count=32):
    loaded_data_list = []
    for i in range(1, data_count):
        loaded_data = wav16khz2mfcc(path_prefix + str(i) + "/")
        loaded_data_list.append(loaded_data)
    return loaded_data_list


def extract_data_from_loaded(data_to_extract):
    return [data.values() for data in data_to_extract]


def prepare_data(data_to_prepare):
    return [np.vstack(data) for data in data_to_prepare]


class PersonGmm(object):
    __slots__ = ['mean', 'cov', 'weight']

    def __init__(self, mean, cov, weight):
        self.cov = cov
        self.mean = mean
        self.weight = weight


def generate_gmms(data_to_train, count_of_gausses_models):
    persons_gmms = []

    for person in data_to_train:
        mean_values_of_gaussians = person[randint(1, len(person), count_of_gausses_models)]
        covariance_matrices_person = [np.var(person, axis=0)] * count_of_gausses_models
        weight_of_gauss_in_gmm = np.ones(count_of_gausses_models) / count_of_gausses_models

        persons_gmms.append(PersonGmm(mean_values_of_gaussians, covariance_matrices_person, weight_of_gauss_in_gmm))
    return persons_gmms


def train_gmms(train_on_data, gmm_list, iterations=30):
    for i in range(iterations):
        for j in range(len(gmm_list)):
            W, M, C, TTL = train_gmm(train_on_data[j], gmm_list[j].weight, gmm_list[j].mean,
                                     gmm_list[j].cov)
            gmm_list[j].weight = W
            gmm_list[j].cov = C
            gmm_list[j].mean = M

            print("Iteration: {0}, TTL: {1}, Person{2}.".format(i, TTL, j))
    return gmm_list


def test_gmms(gmms_models, test_data):
    score = []
    ll_values = []
    hit_ratio = 0
    alltest = 0
    for i in range(len(test_data)):
        for name, tst in test_data[i].items():
            del score[:]
            del ll_values[:]
            i, name = get_class_and_name(name)
            score.append(name)
            for gmm in gmms_models:
                llv = logpdf_gmm(tst, gmm.weight, gmm.mean, gmm.cov)
                ll_values.append(sum(llv) + np.log(0.03))
            score.append(np.argmax(ll_values) + 1)
            if score[-1] == i:
                hit_ratio += 1
            score.extend(ll_values)
            print score
            alltest += 1

    overal_ratio = (hit_ratio / float(alltest) * float(100))
    print(overal_ratio)

    return overal_ratio


def get_class_and_name(name):
    p = name.split('/')
    try:
        i = int(p[1])
    except ValueError:
        i = 0

    return i, p[2][:-4]


train_data_with_path = load_data_with_name()
dev_data_with_path = load_data_with_name(path_prefix="dev/")

train_data = extract_data_from_loaded(train_data_with_path)
train_data = prepare_data(train_data)
dim = train_data[0].shape[1]

gmms = generate_gmms(train_data, 20)
trained_gmms = train_gmms(train_data, gmms, iterations=200)
test_gmms(gmms, dev_data_with_path)




