# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 16:29:02 2024

@author: yannt
"""

import numpy as np
from sklearn.cluster import KMeans

class IKMeansMinusPlus:
    """
    Incremental K-Means Minus Plus (IKMeansMinusPlus) algorithm for clustering.

    Parameters:
    -----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of centroids to generate.
    
    max_iter : int, optional, default: 300
        Maximum number of iterations of the k-means algorithm for a single run.
    
    tol : float, optional, default: 1e-4
        Relative tolerance with regards to inertia to declare convergence.
    
    Attributes:
    -----------
    n_clusters : int
        The number of clusters.
    
    max_iter : int
        Maximum number of iterations of the k-means algorithm for a single run.
    
    tol : float
        Relative tolerance with regards to inertia to declare convergence.
    
    labels_ : array, shape (n_samples,)
        Labels of each point.
    
    cluster_centers_ : array, shape (n_clusters, n_features)
        Coordinates of cluster centers.
    """
    
    
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.labels_ = None
        self.cluster_centers_ = None

    def fit(self, X):
        """
        Compute IKMeansMinusPlus clustering.
    
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training instances to cluster.
        
        Returns:
        --------
        self
        """
        # Step 1: Initialize with K-means
        kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', n_init=1)
        kmeans.fit(X)
        self.labels_ = kmeans.labels_
        self.cluster_centers_ = kmeans.cluster_centers_
        
        # Step 2: Initialize variables
        success = 0

        while success <= self.n_clusters / 2:
            # Step 3: Select cluster with largest gain
            si_idx, si_gain = self.select_cluster_with_largest_gain(X)
            if si_idx is None:
                break
            
            # Step 4: Check gain condition
            if np.sum(si_gain < si_gain[si_idx]) > self.n_clusters / 2:
                break

            # Step 5: Select pair of clusters with smallest cost
            si_idx, sj_idx = self.select_pair_of_clusters_with_smallest_cost(X, si_idx)
            if si_idx is None or sj_idx is None:
                break

            # Step 6: Check cost condition
            si_cost = np.zeros(self.n_clusters)
            for j in range(self.n_clusters):
                if j != si_idx and self.labels_[j] != -1:
                    si_points = X[self.labels_ == si_idx]
                    sj_points = X[self.labels_ == j]
                    cost = np.linalg.norm(si_points - sj_points)
                    si_cost[j] = cost

            if np.sum(si_cost < si_cost[si_idx]) > self.n_clusters / 2:
                self.labels_[si_idx] = -1  # Mark as indivisible cluster
                continue

            # Step 7: Save current solution and update
            old_centers = np.copy(self.cluster_centers_)
            self.update_solution(si_idx, sj_idx, X)

            # Step 8: Evaluate new solution and update
            if self.evaluate_new_solution(X, old_centers):
                success += 1
            else:
                self.labels_[si_idx] = -1  # Mark pair as unmatchable
            
            # Step 9: Check success condition
            if success > self.n_clusters / 2:
                break

        return self

    def select_cluster_with_largest_gain(self, X):
        """
        Selects the cluster with the largest gain.
    
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data points.
    
        Returns:
        --------
        max_gain_idx : int
            Index of the cluster with the largest gain.
        gains : array, shape (n_clusters,)
            Array of gains for each cluster.
        """
        # Step 3: Select cluster with largest gain
        gains = np.zeros(self.n_clusters)
        for i in range(self.n_clusters):
            if self.labels_[i] != -1:
                cluster_points = X[self.labels_ == i]
                cluster_center = self.cluster_centers_[i]
                sse = np.sum((cluster_points - cluster_center) ** 2)
                gains[i] = sse
        
        max_gain_idx = np.argmax(gains)
        return max_gain_idx, gains

    def select_pair_of_clusters_with_smallest_cost(self, X, si_idx):
        """
        Selects the pair of clusters with the smallest cost relative to the given cluster.
    
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data points.
        si_idx : int
            Index of the source cluster.
    
        Returns:
        --------
        si_idx : int
            Index of the source cluster.
        min_cost_idx : int
            Index of the target cluster with the smallest cost.
        """
        # Step 5: Select pair of clusters with smallest cost
        si_cost = np.zeros(self.n_clusters)
        for j in range(self.n_clusters):
            if j != si_idx and self.labels_[j] != -1:
                si_points = X[self.labels_ == si_idx]
                sj_points = X[self.labels_ == j]
                cost = np.linalg.norm(si_points - sj_points)
                si_cost[j] = cost

        min_cost_idx = np.argmin(si_cost)
        return si_idx, min_cost_idx

    def update_solution(self, si_idx, sj_idx, X):
        """
        Updates the current solution by modifying the cluster centers.
    
        Parameters:
        -----------
        si_idx : int
            Index of the source cluster.
        sj_idx : int
            Index of the target cluster.
        X : array-like, shape (n_samples, n_features)
            Data points.
        """
        # Step 7: Save current solution and update
        random_point_index = np.random.choice(len(X))
        self.cluster_centers_[sj_idx] = X[random_point_index]
        kmeans = KMeans(n_clusters=self.n_clusters, init=self.cluster_centers_, n_init=1)
        kmeans.fit(X)
        self.labels_ = kmeans.labels_
        self.cluster_centers_ = kmeans.cluster_centers_

    def evaluate_new_solution(self, X, old_centers):
        """
        Evaluates the new solution and decides whether to accept it or not.
    
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data points.
        old_centers : array-like, shape (n_clusters, n_features)
            Cluster centers of the previous solution.
    
        Returns:
        --------
        bool
            True if the new solution is accepted, False otherwise.
        """
        # Step 8: Evaluate new solution and update
        old_sse = np.sum((X - old_centers[self.labels_]) ** 2)
        new_sse = np.sum((X - self.cluster_centers_[self.labels_]) ** 2)
        if new_sse < old_sse:
            return True
        return False

    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.

        Parameters:
        -----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            New data to predict.

        Returns:
        --------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        if self.labels_ is None or self.cluster_centers_ is None:
            raise ValueError("Model has not been trained yet. Please call 'fit' method first.")
        
        distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
        labels = np.argmin(distances, axis=1)
        return labels
    
    
    def fit_predict(self, X):
       """
       Compute cluster centers and predict cluster index for each sample.

       Parameters:
       -----------
       X : array-like, shape (n_samples, n_features)
           Training instances to cluster.

       Returns:
       --------
       labels : array, shape [n_samples,]
           Index of the cluster each sample belongs to.
       """
       self.fit(X)
       return self.labels_

