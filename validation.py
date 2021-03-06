
__author__ = 'Nick'

import csv, os, json
import sys
from random import sample, shuffle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import StratifiedKFold
from datetime import datetime
import pickle
#from multiprocessing import Pool
sys.stdout.flush()
start = datetime.now()

def make_negative(driver, num_drivers = 100, num_traces = 2):
    # Makes a set of negative data, not including the current driver
    negData = np.empty((0, num_features - 1))
    negDrivers = sample(drivers, num_drivers)
    if driver in negDrivers: #remove current driver if he's there
        negDrivers.remove(driver)
    else: # otherwise, remove one at random
        negDrivers = sample(negDrivers, num_drivers - 1)
    # Now get features for these journeys
    for d in negDrivers:
        negTraces = sample(xrange(200), num_traces)
        for t in negTraces:
            negData = np.vstack((negData, np.asarray(features[d][t][1:])))
    return negData

def classify(trainData, trainLabels, testData, n_trees, depth, LRate):
    global importance
    y = np.ones((testData.shape[0],))
    #clf = GradientBoostingRegressor(n_estimators = n_trees, max_depth = depth, learning_rate = LRate)
    clf = RandomForestClassifier(n_estimators=n_trees ,max_depth = depth, n_jobs=-1)
    clf.fit(trainData, trainLabels)
    # y = clf.predict(testData)
    y = clf.predict_proba(testData)[:,1]
    #y = 0.5 * np.ones(testData.shape[0])
    #y = clf.predict_proba(testData)
    #y = y[:,1]
    #feat_import = np.asarray(clf.feature_importances_)
    #importance = np.vstack((importance, feat_import))
    return y

def diff(big,small):
    small = set(small)
    return [aa for aa in big if aa not in small]

def gen_validation(driver, n_trees, depth, LRate):
    thisauc = [] #initialise empty list to store auc scores for this driver
    posData = np.asarray(features[driver]) # get features for this driver
    posData = posData[:,1:] # drop the column with trip number
    posData = posData[posData[:,20] < 100, :] # check for speeds over 100m/s
    negData = make_negative(driver, 101, 2) # make a negative set of 100 drivers (it removes one every time in case the current driver is in the list of 101)
    negData = negData[negData[:,20] < 100, :] # check for speeds over 100 m/s
    posLabels = np.ones((posData.shape[0],))
    negLabels = np.zeros((negData.shape[0],))
    trainData = np.append(posData, negData, axis = 0)
    trainLabels = np.append(posLabels, negLabels, axis = 0)
    skf = StratifiedKFold(trainLabels, n_folds = 5, shuffle=True)
    for train_idx, test_idx in skf:
        y = classify(trainData[train_idx,:], trainLabels[train_idx], trainData[test_idx, :], n_trees, depth, LRate)
        # Combine classifier with cosine similarities
        thisauc = np.append(thisauc, roc_auc_score(trainLabels[test_idx], y))
    return thisauc

def load_features():
    global num_features, features, drivers
    f = open('all_features.csv')
    csv_f = csv.reader(f)
    csv_f.next()
    # build dictionary of features
    drivers = set()
    features = {}
    for row in csv_f:
        driver = row[1]
        drivers.add(driver)
        if driver not in features:
            try:
                temp = [float(row[x]) for x in feature_selection]
            except:
                print driver, [row[x] for x in feature_selection]
            features[driver] = [temp]
        else:
            try:
                temp = [float(row[x]) for x in feature_selection]
            except:
                print driver, [row[x] for x in feature_selection]
            features[driver].append(temp)
    num_features = len(features['1'][1])

    # remove NaN
    for driver in drivers:
        for trip in range(len(features[driver])):
            features[driver][trip] = [0 if x=='NaN' else x for x in features[driver][trip]]
    return drivers, features

# NB I'm not normalising probabilities here, because I do that at the very end of training.
def normalise_probs(table):
    smallest = min(table[:,2])
    table[:,2] = table[:,2] - smallest + 0.1
    largest = max(table[:,2])
    table[:,2] = table[:,2] / largest * 0.99
    return table

def find_auc(driv, n_trees, depth, LRate):
    auc = np.zeros([1,5]) # initialise table for results
    for i, driver in enumerate(driv):
        auc = np.vstack((auc,gen_validation(driver, n_trees, depth, LRate)))
        sys.stdout.write("\rMean and Var AUC so far: %1.5f, %1.5f. Just finished driver %d. (%d/%d)" \
                         % (np.mean(auc[1:,:]), np.var(auc[1:,:]), int(driver), i+1, len(driv)))
        sys.stdout.flush()
    auc = auc[1:,:]
    meanAUC = np.mean(auc)
    varAUC = np.var(auc)
    return auc, meanAUC, varAUC

def grid_search(driv, list_trees, list_depth, list_lr, num_drivers):
    driv = sample(driv, num_drivers)
    shuffle(driv)
    auc_results = np.zeros([len(list_lr), len(list_depth), len(list_trees)])
    lrPos = 0
    i = 0
    for lr in list_lr:
        depthPos = 0
        for depth in list_depth:
            estPos = 0
            for n_trees in list_trees:
                thisaucgrid, thisMeanAUC, thisVarAUC = find_auc(driv, n_trees, depth, lr)
                auc_results[lrPos, depthPos, estPos] = thisMeanAUC
                estPos += 1
                print "\nRESULT: Mean AUC = %1.5f, Var AUC = %1.5f for %d estimators of max depth %d with learning rate %1.3f\n" % \
                    (thisMeanAUC, thisVarAUC, n_trees, depth, lr)
                runningmean = np.divide(np.cumsum(np.mean(thisaucgrid,1)), range(1,(num_drivers+1)),dtype='float')
                ax = plt.gca()
                color_cycle = ax._get_lines.color_cycle
                next_color = next(color_cycle)
                plt.plot(range(num_drivers), np.mean(thisaucgrid,1), '-o', label="Config %d"%i, color=next_color)
                plt.plot(range(num_drivers), runningmean, '--o',label="%d running mean"%i, color=next_color)
                # plt.title("\nMean AUC = %1.5f, Var AUC = %1.5f for %d estimators of max depth %d\n" % \
                #     (thisMeanAUC, thisVarAUC, n_trees, depth))
                # handles, labels = ax.get_legend_handles_labels()
                # ax.legend(handles, labels)
                i += 1
            depthPos += 1
        lrPos += 1
    return auc_results

if __name__ == '__main__':
    #freeze_support()
    global drivers, features, feature_selection
    feature_selection = [0, 1, 2, 3, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
    feature_selection = [(x + 2) for x in feature_selection] # col 0 is a counter, col 1 is driver, col 2 is tripnumber
    drivers, features = load_features()
    importance = np.asarray(range(num_features - 1), dtype = np.int) + 1
    driv = sorted(drivers)
    # Choose values for grid search
    try_n_trees = [500]
    try_depth = [4,5,6,7,8,9,10]
    try_lr = [0.05]
    auc_results = grid_search(driv, try_n_trees, try_depth, try_lr, 500)
    auc = np.zeros([1,5]) # initialise table for results
    shuffle(driv)
    # for driver in driv:
    #    auc = np.vstack((auc,gen_validation(driver, 200, 5, 0.1)))
    #    print "Mean AUC so far: %1.4f. Just finished driver %d." % (np.mean(auc[1:,:]), int(driver))
    auc = auc[1:,:] # drop first row of auc (initialised to zero)
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    plt.savefig("validation_auc.png")
    print 'Done, elapsed time: %s' % str(datetime.now() - start)
