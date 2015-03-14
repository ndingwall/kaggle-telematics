__author__ = 'Nick'

import csv, os, json
import sys
from random import sample
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
from multiprocessing import Pool
sys.stdout.flush()
start = datetime.now()

def make_negative(driver, num_drivers = 100, num_traces = 2):
    # Makes a set of negative data, not including the current driver
    negData = np.empty((0, num_features - 1))
    negDrivers = sample(drivers, num_drivers)
    if driver in negDrivers:
        negDrivers.remove(driver)
    else:
        negDrivers = sample(negDrivers, num_drivers - 1)
    for d in negDrivers:
        negTraces = sample(xrange(200), num_traces)
        for t in negTraces:
            negData = np.vstack((negData, np.asarray(features[d][t][1:])))
    return negData

def classify(trainData, trainLabels, testData):
    global importance
    y = np.zeros((testData.shape[0],2))
    #clf = GradientBoostingRegressor(n_estimators=125, max_depth = 3, learning_rate = 0.075)
    # clf = RandomForestRegressor(n_estimators=125, max_depth = 4, n_jobs=-1)
    # clf.fit(trainData, trainLabels)
    # y = clf.predict(testData)
    clf = RandomForestClassifier(n_estimators=500, oob_score=True ,max_depth = 5, n_jobs=-1)
    clf.fit(trainData, trainLabels)
    # y = clf.predict(testData)
    y1 = clf.predict_proba(testData)[:,1]
    y2 = clf.oob_decision_function_
    y2 = y2[trainLabels == 1,1]
    y[:,0] = y1
    y[:,1] = y2
    #y = clf.predict_proba(testData)
    #y = y[:,1]
    feat_import = np.asarray(clf.feature_importances_)
    # importance = np.vstack((importance, feat_import))
    return y

def gen_training_data_and_train(driver):
    global output
    print "Working on driver %d" % (int(driver))
    y = np.ones([200,2]) * 0.95
    posData = np.asarray(features[driver])
    posData = posData[:,1:]
    ignoredTrips = posData[:,20] > 100
    posData = posData[-ignoredTrips, :] # check for speeds over 100m/s
    negData = make_negative(driver, 101, 2)
    negData = negData[negData[:,20] < 100, :] # check for speeds over 100m/s
    posLabels = np.ones((posData.shape[0],))
    negLabels = np.zeros((negData.shape[0],))
    testData = posData
    #y = np.ones((testData.shape[0],))
    trainData = np.append(posData, negData, axis = 0)
    trainLabels = np.append(posLabels, negLabels, axis = 0)
    y[-ignoredTrips, :] = classify(trainData, trainLabels, testData)
    driverCol = int(driver) * np.ones((len(y)))
    tripCol = np.asarray([int(item[0]) for item in features[driver]])
    resultsTemp = np.zeros((4,len(y)))
    resultsTemp[0,:] = driverCol
    resultsTemp[1,:] = tripCol
    resultsTemp[2,:] = y[:,0]
    resultsTemp[3,:] = y[:,1]
    resultsTable = resultsTemp.T
    #output += write_kaggle(driver, features[driver], y)
    return resultsTable

def load_features():
    global num_features, features, drivers
    # load csv
    #f = open('driverFeaturesFixed.csv')
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
            #features[driver].append(row[2:])
    num_features = len(features['1'][1])

    # remove NaN
    for driver in drivers:
        for trip in range(len(features[driver])):
            features[driver][trip] = [0 if x=='NaN' else x for x in features[driver][trip]]
    return drivers, features

def make_sub(output):
    #returnstring = ""
    for i in range(len(output)):
        if output[i][0] == 0:
            returnstring = "driver_trip,prob\n"
        else:
            returnstring += "%d_%d,%.8f\n" % (output[i][0], output[i][1], output[i][2])
    return returnstring

def normalise_probs(table):
    smallest = min(table[:,2])
    table[:,2] = table[:,2] - smallest + 0.1
    largest = max(table[:,2])
    table[:,2] = table[:,2] / largest * 0.99
    return table

if __name__ == '__main__':
    #freeze_support()
    global drivers, features, feature_selection
    #feature_selection = [0, 1, 2, 3, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    feature_selection = [0, 1, 2, 3, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
    feature_selection = [(x + 2) for x in feature_selection]
    drivers, features = load_features()
    importance = np.asarray(range(num_features - 1), dtype = np.int) + 1
    # make predictions
    output = ""
    driv = sorted(drivers)
    driv = sample(driv, 50)
    driv = ['1138']
    #p = Pool(4)
    #p.map(gen_training_data_and_train, driv)
    results = np.zeros((1,4))
    for driver in driv:
        results = np.vstack((results,gen_training_data_and_train(driver)))
    # make submissions
    submission_id = datetime.now().strftime("%Y_%d_%B_%H_%M")
    text_file = open("nick-code-{0}.csv".format(submission_id), "w")
    text_file.write(make_sub(results[:,[0,1,2]]))
    text_file.close
    text_file_2 = open("nick-code-{0}-OOB.csv".format(submission_id), "w")
    text_file_2.write(make_sub(results[:,[0,1,3]]))
    text_file_2.close
    np.savetxt("importance-{0}.csv".format(submission_id), importance, delimiter=",")
    print 'Done, elapsed time: %s' % str(datetime.now() - start)
