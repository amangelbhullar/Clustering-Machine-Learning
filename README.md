# K-Means-clustering

objective- identify pattern of customers in a mall.

### Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd


### Importing the dataset

dataset = pd.read_csv("Mall_Customers.csv")

x = dataset.iloc[:,[3,4]].values

### Using the elbow method to find the optimal number of clusters (use k-means++ to avoid falling into number of clusters trap)

from sklearn.cluster import KMeans

wcss = []

for i in range(1,11):

  kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 1)
  
  kmeans.fit(x)
  
  wcss.append(kmeans.inertia_)
  
plt.plot(range(1, 11), wcss)

plt.title("The elbow method")

plt.xlabel("nuber of cluster")

plt.ylabel("WCSS")

plt.show()


