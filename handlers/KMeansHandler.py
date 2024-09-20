import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from .AbstractClusterHandler import AbstractClusterHandler
import logging
import sys
from typing import Tuple

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

class KMeansHandler(AbstractClusterHandler):
    """KMeans clustering handler with automatic cluster number selection using elbow method and Silhouette score."""
    def __init__(self, train_X, train_y, test_X, test_y, max_clusters=20, mergeCluster=True, plt=False, pltNo=0):
        super().__init__(train_X, train_y, test_X, test_y, mergeCluster, plt, pltNo)
        self.max_clusters = max_clusters

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
        """Cluster a single column of data using KMeans with automatic cluster number selection."""
        # Extract data
        spl = (self.train_y if is_target else self.train_X)[column_name].to_numpy().reshape(-1, 1)
        test_spl = (self.test_y if is_target else self.test_X)[column_name].to_numpy().reshape(-1, 1)

        # Find the optimal number of clusters
        n_clusters = self._find_optimal_clusters(spl)

        # Perform KMeans clustering with the optimal number of clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cls1 = kmeans.fit_predict(spl)

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
        """Find the optimal number of clusters using the Elbow Method and Silhouette Score."""
        inertias = []
        silhouette_scores = []
        n_clusters_range = range(2, min(self.max_clusters + 1, len(data)))

        for n_clusters in n_clusters_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(data)
            inertias.append(kmeans.inertia_)
            if n_clusters > 2:
                silhouette_scores.append(silhouette_score(data, kmeans.labels_))

        # Normalize inertias and silhouette scores
        inertias = (inertias - np.min(inertias)) / (np.max(inertias) - np.min(inertias))
        silhouette_scores = (silhouette_scores - np.min(silhouette_scores)) / (np.max(silhouette_scores) - np.min(silhouette_scores))

        # Combine inertia and silhouette score
        combined_score = inertias[1:] - silhouette_scores

        # Find the elbow point
        optimal_clusters = np.argmin(combined_score) + 3  # +3 because we started from 2 clusters and silhouette from 3

        return optimal_clusters

    def _calculate_cluster_stats(self, df, cluster_col):
        """Calculate mean and std for each cluster."""
        for cluster_id, cluster_data in df.groupby(cluster_col):
            df.loc[df[cluster_col] == cluster_id, 'Cluster_Mean'] = cluster_data['Data'].mean()

            std = cluster_data['Data'].std(ddof=1)
            if std == 0: std = 1
            df.loc[df[cluster_col] == cluster_id, 'Cluster_Std'] = cluster_data['Data'].std(ddof=1)

    def _membership(self, x, sampleMean, sampleDevi):
        """Calculate Gaussian membership."""
        return np.exp(-((x - sampleMean)**2 / (2 * (sampleDevi**2))))