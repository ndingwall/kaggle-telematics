
__author__ = 'Nick'

import csv, os, json
import sys
from random import sample, shuffle
import numpy as np
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
    negCosines = np.empty((0,1))
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
            negCosines = np.vstack((negCosines, cosine_similarities[d][t - 1][1]))
    return negData, negCosines

def classify(trainData, trainLabels, testData, n_trees, depth, LRate):
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

def gen_validation(driver, n_trees, depth, LRate):
    cosine = np.asarray(cosine_similarities[driver])
    thisauc = [] #initialise empty list to store auc scores for this driver
    posData = np.asarray(features[driver]) # get features for this driver
    posData = posData[:,1:] # drop the column with trip number
    posData = posData[posData[:,20] < 100, :] # check for speeds over 100m/s
    negData = make_negative(driver, 101, 2) # make a negative set of 100 drivers (it removes one every time in case the current driver is in the list of 101)
    negData, negCosines = negData[negData[:,20] < 100, :] # check for speeds over 100 m/s
    posLabels = np.ones((posData.shape[0],))
    negLabels = np.zeros((negData.shape[0],))
    trainData = np.append(posData, negData, axis = 0)
    trainLabels = np.append(posLabels, negLabels, axis = 0)
    skf = StratifiedKFold(trainLabels, n_folds = 5, shuffle=True)
    for train_idx, test_idx in skf:
        y1 = classify(trainData[train_idx,:], trainLabels[train_idx], trainData[test_idx, :], n_trees, depth, LRate)
        # Combine classifier with cosine similarities
        y2 = cosine[:,1]
        y = (9 * y1 + 1 * y2[test_idx]) / 10
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

def load_cosine_similarities():
    pkl_file = open('cosines.pkl', 'rb')
    cosine_similarities = pickle.load(pkl_file)
    pkl_file.close()
    drivers = cosine_similarities.keys()
    drivers.sort()
    for driver in drivers:
        cosine_similarities[driver][1] = [cosine_similarities[driver][0], cosine_similarities[driver][1]]
        cosine_similarities[driver].pop(0)
    return cosine_similarities

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
    cosine_similarities = load_cosine_similarities()
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
    shuffle(driv)
    for driver in driv:
       auc = np.vstack((auc,gen_validation(driver, 200, 5, 0.1)))
       print "Mean AUC so far: %1.4f. Just finished driver %d." % (np.mean(auc[1:,:]), int(driver))
    auc = auc[1:,:] # drop first row of auc (initialised to zero)
    print 'Done, elapsed time: %s' % str(datetime.now() - start)
