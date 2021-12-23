# k_nearest_neighbors.py: Machine learning implementation of a K-Nearest Neighbors classifier from scratch.
#
# Submitted by: [enter your full name here] -- [enter your IU username here]
#
# Based on skeleton code by CSCI-B 551 Fall 2021 Course Staff

import numpy as np
from utils import euclidean_distance, manhattan_distance


class KNearestNeighbors:
    """
    A class representing the machine learning implementation of a K-Nearest Neighbors classifier from scratch.

    Attributes:
        n_neighbors
            An integer representing the number of neighbors a sample is compared with when predicting target class
            values.

        weights
            A string representing the weight function used when predicting target class values. The possible options are
            {'uniform', 'distance'}.

        _X
            A numpy array of shape (n_samples, n_features) representing the input data used when fitting the model and
            predicting target class values.

        _y
            A numpy array of shape (n_samples,) representing the true class values for each sample in the input data
            used when fitting the model and predicting target class values.

        _distance
            An attribute representing which distance metric is used to calculate distances between samples. This is set
            when creating the object to either the euclidean_distance or manhattan_distance functions defined in
            utils.py based on what argument is passed into the metric parameter of the class.

    Methods:
        fit(X, y)
            Fits the model to the provided data matrix X and targets y.

        predict(X)
            Predicts class target values for the given test data matrix X using the fitted classifier model.
    """

    def __init__(self, n_neighbors = 5, weights = 'uniform', metric = 'l2'):
        # Check if the provided arguments are valid
        if weights not in ['uniform', 'distance'] or metric not in ['l1', 'l2'] or not isinstance(n_neighbors, int):
            raise ValueError('The provided class parameter arguments are not recognized.')

        # Define and setup the attributes for the KNearestNeighbors model object
        self.n_neighbors = n_neighbors
        self.weights = weights
        self._X = None
        self._y = None
        self._distance = euclidean_distance if metric == 'l2' else manhattan_distance

    def fit(self, X, y):
        """
        Fits the model to the provided data matrix X and targets y.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

        Returns:
            None.
        """
        self._X = X
        self._y = y


        #raise NotImplementedError('This function must be implemented by the student.')

    def predict(self, X):
        """
        Predicts class target values for the given test data matrix X using the fitted classifier model.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the test data.

        Returns:
            A numpy array of shape (n_samples,) representing the predicted target class values for the given test data.
        """

        # result = np.array([])
        # for i in range(len(X)):
        #     dist_array = np.array([])
        #     for j in range(len(self._X)):
        #         dist = self._distance(X[i], self._X[j])
        #         dist_array = np.append(dist_array,dist)
        #     indices_list = np.argpartition(dist_array, self.n_neighbors)[:self.n_neighbors]
        #     y_vals = self._y[indices_list]
        #     result = np.append(result, np.argmax(np.bincount(y_vals)))
        # return result

        result = np.array([])
        #unique_vals = np.unique(self._y)
        for i in range(len(X)):
            dist_array = np.array([])
            unique_y_vals = np.unique(self._y)
            for j in range(len(self._X)):
                dist = self._distance(X[i], self._X[j])
                dist_array = np.append(dist_array, dist)
            idx = np.argpartition(dist_array, self.n_neighbors)[:self.n_neighbors]
            if self.weights == "uniform":
                result = np.append(result, np.amax(self._y[idx]))
            else:
                #print(dist_array)
                dist_arr = np.where(dist_array[idx] == 0, 0.001, dist_array[idx])
                inv_dist = np.reciprocal(dist_arr)
                #print(inv_dist)
                result = np.append(result, unique_y_vals[np.argmax(np.bincount(np.searchsorted(unique_y_vals, self._y[idx]), inv_dist))])
        return result


        # result = np.array([])
        # unique_vals = np.unique(self._y)
        # for i in range(len(X)):
        #     dist_array, inv_dist_array = np.array([]), np.array([])
        #     for j in range(len(self._X)):
        #         dist = self._distance(X[i], self._X[j])
        #         dist_array = np.append(dist_array, dist)
        #         #inv_dist_array = np.append(inv_dist_array, (1/dist))
        #
        #
        #     indices_list = np.argpartition(dist_array, self.n_neighbors)[:self.n_neighbors]
        #     y_vals = self._y[indices_list]
        #     if self.weights == "uniform":
        #         result = np.append(result, np.argmax(np.bincount(y_vals)))
        #     else:
        #         inv_dist_vals = inv_dist_array[indices_list]
        #         sum_arr = np.bincount(np.searchsorted(unique_vals, y_vals), inv_dist_vals)
        #         result = np.append(result, unique_vals[np.argmax(sum_arr)])
        # return result


        #raise NotImplementedError('This function must be implemented by the student.')
