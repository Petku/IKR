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

dim = train_persons[0].shape()[1]

