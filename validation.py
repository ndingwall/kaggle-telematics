
__author__ = 'Nick'

import csv, os, json
import sys
from random import sample, shuffle
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
from datetime import datetime
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
        negTraces = sample(xrange(100), num_traces)
        for t in negTraces:
            negData = np.vstack((negData, np.asarray(features[d][t][1:])))
    return negData

def classify(trainData, trainLabels, testData, n_trees, depth, LRate):
    global importance
    y = np.ones((testData.shape[0],))
    #clf = GradientBoostingRegressor(n_estimators = n_trees, max_depth = depth, learning_rate = LRate)
    clf = RandomForestRegressor(n_estimators=n_trees, max_depth = depth)
    clf.fit(trainData, trainLabels)
    y = clf.predict(testData)
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
    negData = make_negative(driver, 101, 2) # make a negative set of 100 drivers (it removes one every time in case the current driver is in the list of 101)
    tripsShuffled = range(200)
    shuffle(tripsShuffled) # shuffle trip numbers.
    for fold in range(5):
        # Admin
        testIdx = tripsShuffled[fold * 40:fold * 40 + 40] # Choose indexes for test set
        trainIdx = diff(tripsShuffled, testIdx) # find remaining indexes for training set
        trainData = np.append(posData[trainIdx,:], negData[trainIdx,:], axis = 0) # build train set
        testData = np.append(posData[testIdx,:], negData[testIdx, :], axis = 0) # build test set
        #testData = np.append(posData[testIdx,:], negData[testIdx[1:3], :], axis = 0) # build test set
        trainLabels = np.append(np.ones(len(trainIdx)), np.zeros(len(trainIdx))) # write train labels
        testLabels = np.append(np.ones(len(testIdx)), np.zeros(len(testIdx))) # write test labels
        #testLabels = np.append(np.ones(len(testIdx)), np.zeros(2)) # write test labels
        # THE MAGIC HAPPENS HERE!
        y = classify(trainData, trainLabels, testData, n_trees, depth, LRate) # train the classifier and return predictions
        #
        thisauc = np.append(thisauc, roc_auc_score(testLabels, y)) # calculate AUC and add to a list
    return thisauc

def load_features():
    global num_features, features, drivers
    # load csv
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
            #features[driver] = [row[2:]]
            temp = [row[x] for x in feature_selection]
            features[driver] = [temp]
        else:
            temp = [row[x] for x in feature_selection]
            features[driver].append(temp)
            #features[driver].append(row[2:])
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
    for driver in driv:
        auc = np.vstack((auc,gen_validation(driver, n_trees, depth, LRate)))
        #print "Mean AUC so far: %1.4f. Just finished driver %d." % (np.mean(auc[1:,:]), int(driver))
    auc = auc[1:,:]
    meanAUC = np.mean(auc)
    return meanAUC

def grid_search(driv, list_trees, list_depth, list_lr, num_drivers):
    driv = sample(driv, num_drivers)
    shuffle(driv)
    auc_results = np.zeros([len(list_lr), len(list_depth), len(list_trees)])
    lrPos = 0
    for lr in list_lr:
        depthPos = 0
        for depth in list_depth:
            estPos = 0
            for n_trees in list_trees:
                thisAUC = find_auc(driv, n_trees, depth, lr)
                auc_results[lrPos, depthPos, estPos] = thisAUC
                estPos += 1
                print "Mean AUC = %1.4f for %d estimators of max depth %d with learning rate %1.3f" % (thisAUC, n_trees, depth, lr)
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
    # try_n_trees = [50, 75, 100, 125, 150]
    # try_depth = [2,3,4]
    # try_lr = [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25]
    # auc_results = grid_search(driv, try_n_trees, try_depth, try_lr, 50)
    auc = np.zeros([1,5]) # initialise table for results
    for driver in driv:
       auc = np.vstack((auc,gen_validation(driver, 200, 3, 0.1)))
       print "Mean AUC so far: %1.4f. Just finished driver %d." % (np.mean(auc[1:,:]), int(driver))
    auc = auc[1:,:] # drop first row of auc (initialised to zero)
    #np.save('auc_results_March_11', auc_results) # save AUC file in npy format
    print 'Done, elapsed time: %s' % str(datetime.now() - start)
    #print np.mean(auc)
    #print np.var(auc)