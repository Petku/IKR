from __future__ import print_function

import matplotlib.pyplot as plt
from ikrlib import png2fea, logpdf_gauss, train_gauss, train_gmm, logpdf_gmm
import scipy.linalg
import numpy as np
from numpy.random import randint
from time import time
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

train_persons = []
train_persons_classes = []
for number in range(1, 32):
    train_person = png2fea('train/' + str(number) + '/').values()
    train_persons.extend(train_person)
    train_persons_classes.extend([number] *len(train_person))
    #train_persons[number-1] = np.vstack(train_persons[number-1])

test_persons = []
test_persons_classes = []
test_filenames = []
for number in range(1, 32):
    tst = png2fea('dev/' + str(number) + '/')
    test_filenames.extend(tst.keys())
    test_person = tst.values()
    test_persons.extend(test_person)
    test_persons_classes.extend([number] *len(test_person))
    #train_persons[number-1] = np.vstack(train_persons[number-1])

w = 80
h = 80

train_persons = np.array(train_persons)
dim = train_persons.shape[1]
print("Total dataset size:")
print("n_samples: %d" % len(train_persons))
print("n_features: %d" % dim)
print("n_classes: %d" % 31)

# #############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 64

print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, train_persons.shape[0]))
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(train_persons)
print("done in %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
train_persons_pca = pca.transform(train_persons)
test_persons_pca = pca.transform(test_persons)
print("done in %0.3fs" % (time() - t0))


# #############################################################################
# Train a SVM classification model

print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced', probability=True), param_grid, cv=5)
#clf = MLPClassifier(hidden_layer_sizes=(1024,), batch_size=256, verbose=True, early_stopping=True)
clf = clf.fit(train_persons_pca, train_persons_classes)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
#print(clf.best_estimator_)


# #############################################################################
# Quantitative evaluation of the model quality on the test set

print("Predicting people's names on the test set")
t0 = time()

test_classes_Predictions = clf.predict(test_persons_pca)
print("done in %0.3fs" % (time() - t0))

hit_ratio = 0
all_tests = len(test_classes_Predictions)

pred_proba = clf.predict_log_proba(test_persons_pca)
for file, expected, returned, log_prob in zip(test_filenames, test_persons_classes, test_classes_Predictions, pred_proba):
    print(file, returned, *log_prob, sep=', ')
    if expected == returned:
        hit_ratio+=1

print(hit_ratio/float(all_tests))

