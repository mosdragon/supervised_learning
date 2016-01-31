# coding: utf-8
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.cross_validation import KFold
import data_util as util

# Your data
FILENAME = 'example/data.csv'

# Cross Validation Params
N_FOLDS = 10
PERCENTAGE_TRAINING = 0.70

# KNN Params
num_neighbors = 4
algorithm_used = 'brute'  # ['brute', 'auto']

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

    # Create KNN classifier
    classifier = KNN(n_neighbors=num_neighbors, algorithm=algorithm_used)
    classifier.fit(trnX, trnY)

    # Get score using rest of training data, which is our validation set
    score = classifier.score(vldX, vldY)
    scores.append(score)

    print "Iteration: {}\tScore: {}".format(n_iteration + 1, score)

    # Keep track of the best classifier
    if score > highest_score:
        best_classifier = classifier
        highest_score = score

avg_score = np.mean(scores)
print "AVG SCORE: {}".format(avg_score)
print "High Score: {}".format(highest_score)

# Now run on actual test set
classifier = best_classifier
final_score = classifier.score(*test)
print "Final Score: {}".format(final_score)
