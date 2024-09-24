import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from .AbstractClusterHandler import AbstractClusterHandler
import logging
import sys
from typing import Tuple
from scipy.stats import iqr



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

class DBSCANHandler(AbstractClusterHandler):
    """DBSCAN clustering handler with automatic parameter selection."""
    def __init__(self, train_X, train_y, test_X, test_y, eps=None, min_samples=None, mergeCluster=True, plt=False, pltNo=0):
        super().__init__(train_X, train_y, test_X, test_y, mergeCluster, plt, pltNo)
        self.eps = eps
        self.min_samples = min_samples

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
        """Cluster a single column of data using DBSCAN with automatic parameter selection if not provided."""
        # Extract data
        spl = (self.train_y if is_target else self.train_X)[column_name].to_numpy().reshape(-1, 1)
        test_spl = (self.test_y if is_target else self.test_X)[column_name].to_numpy().reshape(-1, 1)

        # Standardize the data
        # scaler = StandardScaler()
        # spl_scaled = scaler.fit_transform(spl)
        # test_spl_scaled = scaler.transform(test_spl)

        # Find optimal parameters if not provided
        if self.eps is None or self.min_samples is None:
            eps, min_samples = self._find_optimal_parameters(spl)
        else:
            eps, min_samples = self.eps, self.min_samples

        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cls1 = dbscan.fit_predict(spl)

        # Handle noise points (labeled as -1 by DBSCAN)
        # cls1[cls1 == -1] = max(cls1) + 1

        # Handle noise points and ensure a minimum number of clusters
        cls1 = self._post_process_clusters(cls1, spl)

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

    def _find_optimal_parameters(self, data: np.ndarray) -> Tuple[float, int]:
        """Find optimal eps and min_samples for DBSCAN using adaptive methods."""
        # Calculate the interquartile range (IQR) of the data
        data_iqr = iqr(data)

        # Set eps based on the IQR
        eps = data_iqr / 2

        # Set min_samples based on data size, but ensure it's not too large
        min_samples = max(int(0.01 * len(data)), 3)
        min_samples = min(min_samples, 10)  # Cap at 10 to avoid too restrictive clustering

        return eps, min_samples
    
    def _post_process_clusters(self, labels, data):
        """Post-process cluster labels to handle noise and ensure a minimum number of clusters."""
        # Handle noise points
        noise_mask = labels == -1
        if np.sum(noise_mask) > 0:
            # Assign noise points to the nearest non-noise cluster
            non_noise_clusters = np.unique(labels[~noise_mask])
            for point in data[noise_mask]:
                distances = [np.min(np.linalg.norm(data[labels == c] - point, axis=1)) for c in non_noise_clusters]
                nearest_cluster = non_noise_clusters[np.argmin(distances)]
                labels[noise_mask & (data == point).all(axis=1)] = nearest_cluster

        # Ensure a minimum of 3 clusters
        unique_clusters = np.unique(labels)
        if len(unique_clusters) < 3:
            # If we have fewer than 3 clusters, split the largest cluster
            cluster_sizes = [np.sum(labels == c) for c in unique_clusters]
            largest_cluster = unique_clusters[np.argmax(cluster_sizes)]
            largest_cluster_data = data[labels == largest_cluster]
            
            # Use K-means to split the largest cluster
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=2, random_state=42)
            sub_labels = kmeans.fit_predict(largest_cluster_data)
            
            # Update the labels
            new_cluster_label = labels.max() + 1
            labels[labels == largest_cluster] = np.where(sub_labels == 0, largest_cluster, new_cluster_label)

        return labels