import numpy as np
import csv
from datetime import datetime
from scipy.spatial.distance import cosine

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
            temp = [float(row[x]) for x in feature_selection]
            features[driver] = [temp]
        else:
            temp = [float(row[x]) for x in feature_selection]
            features[driver].append(temp)
            #features[driver].append(row[2:])
    num_features = len(features['1'][1])

start = datetime.now()


feature_selection = [0, 1, 2, 3, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
feature_selection = [(x + 2) for x in feature_selection]

load_features()

# for trip in features[driver]:
start = datetime.now()
driver = '1'
cosine_differences = np.zeros([200,200])
for j in range(200):
    for i in range(200):
        cosine_differences[j,i] = cosine(features[driver][j],features[driver][i])

print 'Done, elapsed time: %s' % str(datetime.now() - start)

start = datetime.now()
driver = '1'
cosine_similarity = np.dot(features[driver], np.transpose(features[driver]))

cosine_differences = np.zeros([200,200])
for j in range(200):
    for i in range(200):
        cosine_differences[j,i] = cosine(features[driver][j],features[driver][i])

print 'Done, elapsed time: %s' % str(datetime.now() - start)