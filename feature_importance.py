__author__ = 'Nick'

from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt
my_data = genfromtxt('D:/Nick/Documents/git/Tmatics/JamesFeatures/importance-2015_12_March_12_43.csv', delimiter=',')
features = my_data[0,:]
my_data = my_data[1:,:]
means = np.mean(my_data, axis = 0)

fig = plt.figure()
ax1 = fig.add_subplot(111)

for i in range(my_data.shape[0]):
    ax1.scatter(features, my_data[i,:])

plt.show()
#
# fig = plt.figure()
# fig.violinplot(my_data, features, points=80, vert=False, widths=0.7,
#                       showmeans=True, showextrema=True, showmedians=True)
# plt.show()