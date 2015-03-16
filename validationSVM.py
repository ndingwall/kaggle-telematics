
__author__ = 'Nick'

import csv, os, json
import sys
from random import sample, shuffle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import StratifiedKFold
from datetime import datetime
from decimal import Decimal
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
            negData = np.vstack((negData, np.asarray(normalised_features[d][t][1:])))
    return negData

def classifySVM(trainData, trainLabels, testData, n_trees, depth, LRate):
    global importance
    y = np.ones((testData.shape[0],))
    #clf = GradientBoostingRegressor(n_estimators = n_trees, max_depth = depth, learning_rate = LRate)
    clf = SVC(probability=True)
    # clf = RandomForestClassifier(n_estimators=n_trees, oob_score=True ,max_depth = depth, n_jobs=-1)
    clf.fit(trainData, trainLabels)
    # y = clf.predict(testData)
    y = clf.predict_proba(testData)[:,1]
    #y = 0.5 * np.ones(testData.shape[0])
    #y = clf.predict_proba(testData)
    #y = y[:,1]
    #feat_import = np.asarray(clf.feature_importances_)
    #importance = np.vstack((importance, feat_import))
    return y

def classifyLogisticReg(trainData, trainLabels, testData, n_trees, depth, LRate):
    global importance
    y = np.ones((testData.shape[0],))
    #clf = GradientBoostingRegressor(n_estimators = n_trees, max_depth = depth, learning_rate = LRate)
    clf = LogisticRegression()
    # clf = RandomForestClassifier(n_estimators=n_trees, oob_score=True ,max_depth = depth, n_jobs=-1)
    clf.fit(trainData, trainLabels)
    # y = clf.predict(testData)
    y = clf.predict_proba(testData)[:,1]
    #y = 0.5 * np.ones(testData.shape[0])
    #y = clf.predict_proba(testData)
    #y = y[:,1]
    #feat_import = np.asarray(clf.feature_importances_)
    #importance = np.vstack((importance, feat_import))
    return y

def classifyRandomForest(trainData, trainLabels, testData, n_trees, depth, LRate):
    global importance
    y = np.ones((testData.shape[0],))
    #clf = GradientBoostingRegressor(n_estimators = n_trees, max_depth = depth, learning_rate = LRate)
    clf = RandomForestClassifier(n_estimators=n_trees, oob_score=True ,max_depth = depth, n_jobs=-1)
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

def gen_validation(driver, n_trees, depth, LRate, weights):
    thisauc = np.zeros([5,len(weights)]) #initialise empty list to store auc scores for this driver
    fold = 0
    posData = np.asarray(normalised_features[driver]) # get features for this driver
    posData = posData[:,1:] # drop the column with trip number
    posData = posData[posData[:,20] < 100, :] # check for speeds over 100m/s
    negData = make_negative(driver, 201, 2) # make a negative set of 100 drivers (it removes one every time in case the current driver is in the list of 101)
    negData = negData[negData[:,20] < 100, :] # check for speeds over 100 m/s
    posLabels = np.ones((posData.shape[0],))
    negLabels = np.zeros((negData.shape[0],))
    trainData = np.append(posData, negData, axis = 0)
    trainLabels = np.append(posLabels, negLabels, axis = 0)
    skf = StratifiedKFold(trainLabels, n_folds = 5, shuffle=True)
    for train_idx, test_idx in skf:
        yRF = classifyRandomForest(trainData[train_idx,:], trainLabels[train_idx], trainData[test_idx, :], n_trees, depth, LRate)
        ySVM = classifySVM(trainData[train_idx,:], trainLabels[train_idx], trainData[test_idx, :], n_trees, depth, LRate)
        yLR = classifyLogisticReg(trainData[train_idx,:], trainLabels[train_idx], trainData[test_idx, :], n_trees, depth, LRate)
        weight_counter = 0
        for weight in weights:
            y = (weight * yRF + ySVM + yLR) / (weight + 2)
        # Combine classifier with SVM similarities
        #     thisauc = np.append(thisauc, roc_auc_score(trainLabels[test_idx], y))
        #     print thisauc
            thisauc[fold,weight_counter] = roc_auc_score(trainLabels[test_idx], y)
            weight_counter += 1
        fold += 1
    thisauc = np.mean(thisauc, axis = 0)
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

def find_auc(driv, n_trees, depth, LRate, weight):
    auc = np.zeros([1,5]) # initialise table for results
    for driver in driv:
        auc = np.vstack((auc,gen_validation(driver, n_trees, depth, LRate, weight)))
        # print "Mean AUC so far: %1.4f. Just finished driver %d." % (np.mean(auc[1:,:]), int(driver))
    auc = auc[1:,:]
    meanAUC = np.mean(auc)
    return meanAUC

def grid_search(driv, list_trees, list_depth, list_lr, list_weight, num_drivers):
    driv = sample(driv, num_drivers)
    shuffle(driv)
    # auc_results = np.zeros([len(list_lr), len(list_depth), len(list_trees)])
    auc_results = np.zeros([len(list_weight)])
    lrPos = 0
    lr = 0.1
    for weight in list_weight:
        depthPos = 0
        for depth in list_depth:
            estPos = 0
            for n_trees in list_trees:
                thisAUC = find_auc(driv, n_trees, depth, lr, weight)
                auc_results[lrPos, depthPos, estPos] = thisAUC
                estPos += 1
                print "Mean AUC = %1.4f for %d estimators of max depth %d with learning rate %1.3f" % (thisAUC, n_trees, depth, lr)
            depthPos += 1
        lrPos += 1
    return auc_results

def normalise_features(features, driv):
    normalised_features = {}
    sums_of_features = np.zeros([1,num_features])
    mean = np.zeros([1,num_features])
    M2 = np.zeros([1,num_features])
    n = 0
    for driver in driv:
        thesefeatures = np.asarray(features[driver])
        sums_of_features += sum(sums_of_features)
        for trip in range(len(features[driver])):
            n = n + 1
            delta = thesefeatures[trip,:] - mean
            mean = mean + delta/n
            M2 = M2 + delta * (thesefeatures[trip,:] - mean)
    variance = M2/(n-1)
    num_trips = len(driv) * 200
    means_of_features = sums_of_features / num_trips
    for driver in driv:
        thesefeatures = (np.asarray(features[driver]) - means_of_features) / variance
        normalised_features[driver] = thesefeatures
    return normalised_features

if __name__ == '__main__':
    #freeze_support()
    global drivers, features, feature_selection
    feature_selection = [0, 1, 2, 3, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
    feature_selection = [(x + 2) for x in feature_selection] # col 0 is a counter, col 1 is driver, col 2 is tripnumber
    drivers, features = load_features()
    importance = np.asarray(range(num_features - 1), dtype = np.int) + 1
    driv = sorted(drivers)
    normalised_features = normalise_features(features, driv)
    # Choose values for grid search
    # try_n_trees = [50, 75, 100, 125, 150]
    # try_depth = [2,3,4]
    # try_lr = [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25]
    # auc_results = grid_search(driv, try_n_trees, try_depth, try_lr, 50)
    # auc_results = grid_search(driv, [200], [5], [0.1], [0,0.5,1,2,5,10,20], 300)
    weights = [0,3,4,5,8,10,12,10000]
    print "Weights: 1 SVM : n RF   ", weights
    auc = np.zeros([1,len(weights)]) # initialise table for results
    shuffle(driv)
    for driver in driv:
        auc = np.vstack((auc,gen_validation(driver, 200, 5, 0.1, weights)))
        printable = [float(Decimal("%.4f" % e)) for e in np.mean(auc[1:],axis=0)]
        print "Driver %4d, AUC scores:" % int(driver), printable
       # print "Mean AUC so far: %1.4f. Just finished driver %d." % (np.mean(auc[1:,:]), int(driver))
    # auc = auc[1:,:] # drop first row of auc (initialised to zero)
    # print auc_results
    print 'Done, elapsed time: %s' % str(datetime.now() - start)
