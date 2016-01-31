import pandas as pd
import numpy as np
import data_util as util
from sklearn.svm import SVC as SVM
from sklearn.cross_validation import KFold

# Your data
FILENAME = 'example/data.csv'

# Cross Validation Params
N_FOLDS = 10
PERCENTAGE_TRAINING = 0.70

# SVM Params
KERNEL = 'poly'  # Options: 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
DEGREE = 6

df = util.read_file(FILENAME)
train, test = util.split_data(df, PERCENTAGE_TRAINING)


trainX, trainY = train
n_trn, n_attr = trainX.shape

kf = KFold(n_trn, n_folds=N_FOLDS)
scores = []

best_classifier = None
highest_score = 0

for n_iteration, fold_nums in enumerate(kf):

    trn_i, vld_i = fold_nums

    # Create training and validation subsets
    trnX, trnY = trainX[trn_i], trainY[trn_i]
    vldX, vldY = trainX[vld_i], trainY[vld_i]

    # Create boosting classifier
    classifier = SVM(kernel=KERNEL, degree=DEGREE)
    classifier.fit(trnX, trnY)

    # Get score using rest of training data, which is our validation set
    score = classifier.score(vldX, vldY)
    scores.append(score)

    # Keep track of the best classifier
    if score > highest_score:
        best_classifier = classifier
        highest_score = score


# Now run on actual test set
classifier = best_classifier

train_accuracy = classifier.score(*train)
print "Test Accuracy: {}".format(train_accuracy)

test_accuracy = classifier.score(*test)
print "Test Accuracy: {}".format(test_accuracy)
