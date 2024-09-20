import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import silhouette_score
from .AbstractClusterHandler import AbstractClusterHandler
import logging
import sys
from typing import Tuple
import matplotlib.pyplot as plt

# Configure logging
def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Check if the logger already has handlers to avoid duplicate logging
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

logger = setup_logger(__name__)

class AgglomerativeHandler(AbstractClusterHandler):
    """Agglomerative clustering handler with automatic cluster number selection."""
    def __init__(self, train_X, train_y, test_X, test_y, n_clusters=None, max_clusters=10, 
                 linkage='ward', metric='euclidean', mergeCluster=True, plt=False, pltNo=0):
        super().__init__(train_X, train_y, test_X, test_y, mergeCluster, plt, pltNo)
        self.n_clusters = n_clusters
        self.max_clusters = max_clusters
        self.linkage = linkage
        self.metric = metric

    def cluster(self):
        """Main clustering method that processes all columns."""
        train_X_res = []
        train_y_res = []
        test_X_res = []
        test_y_res = []

        assert self.train_X.shape[1] == self.test_X.shape[1], "Train and test data must have the same number of columns"
        assert self.train_y.shape[1] == self.test_y.shape[1], "Train_y and test_y data must have the same number of columns"
        assert self.train_X.columns.tolist() == self.test_X.columns.tolist(), "Train and test data must have the same columns"
        assert self.train_y.columns.tolist() == self.test_y.columns.tolist(), "Train_y and test_y data must have the same columns"

        # Cluster each feature column
        for column in self.train_X.columns:
            temp, test_temp = self._process_column(column, is_target=False)
            train_X_res.append(temp)
            test_X_res.append(test_temp)

        # Combine clustered results for features
        train_X_res = pd.concat(train_X_res, axis=1)
        test_X_res = pd.concat(test_X_res, axis=1)

        # Cluster each target variable
        for column in self.train_y.columns:
            temp, test_temp = self._process_column(column, is_target=True)
            train_y_res.append(temp)
            test_y_res.append(test_temp)

        # Combine clustered results for target variables
        train_y_res = pd.concat(train_y_res, axis=1)
        test_y_res = pd.concat(test_y_res, axis=1)
        
        return train_X_res, train_y_res, test_X_res, test_y_res

    def _process_column(self, column_name, is_target=False):
        """Process a single column for clustering."""
        temp, testTemp, clsDic = self._cluster_column(column_name, is_target)

        # Remove temporary columns
        temp.drop([f"Cluster_{column_name}", "Data", "Cluster_Mean", "Cluster_Std"], axis=1, inplace=True)
        testTemp.drop(["Data"], axis=1, inplace=True)

        if is_target:
            temp.index = self.train_y.index
            testTemp.index = self.test_y.index
        else:
            testTemp.index = self.test_X.index
            temp.index = self.train_X.index

        # Store cluster statistics
        if is_target:
            self.y_cluster_stats[column_name] = clsDic
        else:
            self.feature_cluster_stats[column_name] = clsDic
        return temp, testTemp

    def _cluster_column(self, column_name, is_target=False):
        """Cluster a single column of data using Agglomerative clustering."""
        # Extract data
        spl = (self.train_y if is_target else self.train_X)[column_name].to_numpy().reshape(-1, 1)
        test_spl = (self.test_y if is_target else self.test_X)[column_name].to_numpy().reshape(-1, 1)

        # Standardize the data
        # scaler = StandardScaler()
        # spl_scaled = scaler.fit_transform(spl)
        # test_spl_scaled = scaler.transform(test_spl)

        # Find optimal number of clusters if not provided
        if self.n_clusters is None:
            self.n_clusters = self._find_optimal_clusters(spl)

        # Perform Agglomerative clustering
        agg = AgglomerativeClustering(n_clusters=self.n_clusters, linkage=self.linkage, metric=self.metric)
        cls1 = agg.fit_predict(spl)

        # Create temporary DataFrames
        temp = pd.DataFrame({'Data': spl.reshape(-1), f'Cluster_{column_name}': cls1})
        testTemp = pd.DataFrame({'Data': test_spl.reshape(-1)})

        # Calculate initial cluster statistics
        self._calculate_cluster_stats(temp, f'Cluster_{column_name}')

        # Merge small clusters if enabled
        if self.mergeCluster:
            self._merge_clusters(temp, f'Cluster_{column_name}')

        # Calculate membership functions
        clsDic = self._calculate_memberships(temp, testTemp, f'Cluster_{column_name}', column_name)

        logger.info(f"{column_name} had {len(clsDic)} clusters")
        return temp, testTemp, clsDic

    def _find_optimal_clusters(self, data: np.ndarray) -> int:
        """Find the optimal number of clusters using silhouette score."""
        silhouette_scores = []
        n_clusters_range = range(2, min(self.max_clusters + 1, len(data)))

        for n_clusters in n_clusters_range:
            agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=self.linkage, metric=self.metric)
            cluster_labels = agg.fit_predict(data)
            silhouette_avg = silhouette_score(data, cluster_labels)
            silhouette_scores.append(silhouette_avg)

        # Find the number of clusters with the highest silhouette score
        optimal_clusters = n_clusters_range[np.argmax(silhouette_scores)]

        return optimal_clusters

    def plot_dendrogram(self, data, max_d=None):
        """Plot the dendrogram for the hierarchical clustering."""
        # Compute the dendrogram linkage
        agg = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage=self.linkage, metric=self.metric)
        agg.fit(data)

        # Create linkage matrix
        counts = np.zeros(agg.children_.shape[0])
        n_samples = len(agg.labels_)
        for i, merge in enumerate(agg.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack([agg.children_, agg.distances_, counts]).astype(float)

        # Plot the dendrogram
        plt.figure(figsize=(10, 7))
        dendrogram(linkage_matrix)
        
        if max_d:
            plt.axhline(y=max_d, c='k')
        
        plt.title('Dendrogram')
        plt.xlabel('Sample index')
        plt.ylabel('Distance')
        plt.show()