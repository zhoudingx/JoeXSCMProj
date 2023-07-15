# Native implementation of K-Means algorithm
import sys
import math
import random

#import xlrd
import numpy as np

import pandas as pd
import scipy as sc
from pandas import ExcelWriter
from pandas import ExcelFile
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# Assume you load your data into a DataFrame called 'df'. 
# Each row is a product, and each column is a product characteristic, e.g. sales, price, review_score.
# df = pd.read_csv('your_data.csv')
# df = pd.read_csv('prod_data_input.csv')   # load CSV file
df = pd.read_excel('prod_data_input_Excel.xlsx', sheet_name='prod_data_input' )
print(df)
# It's usually a good idea to standardize the features to have zero mean and unit variance
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Define the number of clusters
k_num = 3

kmeans = KMeans(n_clusters=k_num)
df['cluster'] = kmeans.fit_predict(df_scaled.data)
print(df)
with pd.ExcelWriter("prod_data_output_Excel_Native.xlsx") as writer2:
    df.to_excel(writer2) 

class KMeans:
    def __init__(self, n_clusters=3, max_iters=300, random_state=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        np.random.seed(random_state)

    def fit_predict(self, X):
        # 1. Randomly initialize the centroids
        random_idx = np.random.permutation(X.shape[0])
        centroids = X[random_idx[:self.n_clusters]]

        for _ in range(self.max_iters):
            # 2. Assign each data point to the closest centroid
            labels = self._closest_centroid(X, centroids)

            # 3. Compute new centroids as the mean of the data points assigned to each centroid
            new_centroids = np.array([X[labels == i].mean(axis=0)
                                      for i in range(self.n_clusters)])

            # 4. If the centroids have not changed, then the algorithm has converged
            if np.all(centroids == new_centroids):
                break

            centroids = new_centroids

        return labels

    def _closest_centroid(self, X, centroids):
        # Compute the distance of each data point to each centroid
        distances = self._euclidean_distance(X, centroids)

        # Return the index of the closest centroid for each data point
        return np.argmin(distances, axis=1)

    @staticmethod
    def _euclidean_distance(X, centroids):
        return np.sqrt(((X[:, np.newaxis] - centroids) ** 2).sum(axis=2))

"""
y_kmeans = kmeans.predict(X)

import numpy as np

def kmeans(X, K):
    centroids = X[np.random.choice(X.shape[0], K, replace=False)]
    while True:
        C = np.argmin(np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2)), axis=0)
        new_centroids = np.array([X[C == k].mean(axis=0) for k in range(K)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return C

X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])
kmeans(X, 2)

"""