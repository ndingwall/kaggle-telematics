import numpy as np
import csv
from datetime import datetime
from random import sample
from scipy.spatial.distance import cosine
import pickle

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

def normalise_features(feature_list):
    m = np.mean(feature_list)
    v = np.var(feature_list)
    features_out = (feature_list - m) / v
    return features_out

# for trip in features[driver]:
start = datetime.now()
# drivers = sample(drivers, 10)
cosines = {}
drivrs = sorted(drivers)
for driver in drivrs:
    temp_features = np.asarray(features[driver])
    for feature in range(1, num_features):
        temp_features[:,feature] = normalise_features(temp_features[:,feature])
    cosine_differences = np.zeros([200,200])
    for j in range(200):
        for i in range(200):
            # cosine_differences[j,i] = cosine(features[driver][j],features[driver][i])
            cosine_differences[j,i] = cosine(temp_features[j,1:],temp_features[i,1:])
        cosine_differences[j,:].sort()
        mean_diff = np.mean(cosine_differences[j,1:11])
        if driver not in cosines:
            cosines[driver] = [j, mean_diff]
        else:
            cosines[driver].append([j, mean_diff])
    print 'Done driver %s, elapsed time: %s' % (driver, str(datetime.now() - start))


# start = datetime.now()
# driver = '1'
# cosine_similarity = np.dot(features[driver], np.transpose(features[driver]))
#
# cosine_differences = np.zeros([200,200])
# for j in range(200):
#     for i in range(200):
#         cosine_differences[j,i] = cosine(features[driver][j],features[driver][i])
# print 'Done, elapsed time: %s' % str(datetime.now() - start)
# norms = np.linalg.norm(features[driver], 2, axis = 1)
# norms_repeated = np.tile(norms, (200,1))
# normaliser = norms_repeated * np.transpose(norms_repeated)
# cosine_differences / normaliser

output = open('cosines.pkl', 'wb')
pickle.dump(cosines, output)
output.close()

# pkl_file = open('cosines.pkl', 'rb')
# mydict2 = pickle.load(pkl_file)
# pkl_file.close()
