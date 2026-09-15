"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np

import utils
from utils import euclidean_dist_squared


class KNN:
    X = None
    y = None

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X  # just memorize the training data
        self.y = y

    def predict(self, X_hat):
        #Calculating distances
        distances = euclidean_dist_squared(self.X, X_hat) 

        #Setting up out matrix
        numTest = X_hat.shape[0]
        y_hat = np.zeros(numTest, dtype=self.y.dtype)

        for i in range(numTest):
            #Indices of k nearest for current test example
            indices = np.argsort(distances[:, i])[:self.k]

            y_hat[i] = utils.mode(self.y[indices])
        
        return y_hat
        

