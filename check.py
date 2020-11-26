import csv
import itertools
import pickle
from cosum import k_means
import matplotlib.pyplot as plt
import csv
import sys 
with open('training/Objects.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
import numpy as np
from numpy import genfromtxt
from sklearn.datasets.samples_generator import make_blobs
my_data = genfromtxt('training/data.csv', delimiter=',')
newData = []

for i in range(len(my_data)):
    cash = []
    for k in range(len(my_data[i])):
        cash.append(float(my_data[i][k]))
    newData.append(cash)

kmeans = k_means(3,max_iterations=100000)
kmeans.fit(newData,metric="euclidean")
centroid = kmeans.centroids
y_means = np.array(kmeans.y_means)

print(len(y_means))
X = np.array(newData)

centroid_x = np.array(centroid)

plt.scatter(X[:, 0], X[:, 1], c=y_means, s=50, cmap='viridis')

plt.scatter(centroid_x[:, 0], centroid_x[:, 1], c='black', s=200, alpha=0.5)
plt.show()