# coding: utf-8
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.cross_validation import KFold
import data_util as util


# Your data
FILENAME = 'example/data.csv'

# DecisionTreeClassifier params
MAX_DEPTH_RATIO = 0.1  # 0.1 %
MIN_SAMPLES_SPLIT_RATIO = 0.005
MIN_SAMPLES_LEAF_RATIO = 0.005

SPLITTER = 'best'  # ['best', 'random'], default 'best'
CRITERION = 'entropy'  # ['entropy', 'GINI'], default 'GINI'
PRESORT = True  # default False

# Cross Validation Params
N_FOLDS = 10
PERCENTAGE_TRAINING = 0.70


# Main

df = util.read_file(FILENAME)

train, test = util.split_data(df, PERCENTAGE_TRAINING)


trainX, trainY = train
n_trn, n_attr = trainX.shape

MAX_DEPTH = max(int(n_trn * MAX_DEPTH_RATIO), 1) # Max depth must be at least 1
MIN_SAMPLES_SPLIT = max(int(n_trn * MIN_SAMPLES_SPLIT_RATIO), 2)  # default 2
MIN_SAMPLES_LEAF = max(int(n_trn * MIN_SAMPLES_LEAF_RATIO), 1)  # default 1

kf = KFold(n_trn, n_folds=N_FOLDS)
scores = []

best_classifier = None
highest_score = 0

for n_iteration, fold_nums in enumerate(kf):

    trn_i, vld_i = fold_nums

    # Create training and validation subsets
    trnX, trnY = trainX[trn_i], trainY[trn_i]
    vldX, vldY = trainX[vld_i], trainY[vld_i]

    # Create Decision Tree classifier
    classifier = tree.DecisionTreeClassifier(criterion='entropy', max_depth=MAX_DEPTH,
        splitter=SPLITTER, min_samples_leaf=MIN_SAMPLES_LEAF,
        min_samples_split=MIN_SAMPLES_SPLIT)

    classifier.fit(trnX, trnY)

    # Get score using rest of training data, which is our validation set
    score = classifier.score(vldX, vldY)
    scores.append(score)
    print "Iteration: {}\tScore: {}".format(n_iteration + 1, score)

    # Keep track of the best classifier
    if score > highest_score:
        best_classifier = classifier
        highest_score = score


# Now run on actual test set
classifier = best_classifier
final_score = classifier.score(*test)
print "Final Score: {}".format(final_score)
