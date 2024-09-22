import numpy as np
import pandas as pd
from sklearn.cluster import Birch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from .AbstractClusterHandler import AbstractClusterHandler
import logging
import sys
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

class BIRCHHandler(AbstractClusterHandler):
    """BIRCH clustering handler."""
    def __init__(self, train_X, train_y, test_X, test_y, 
                 threshold=0.01, branching_factor=50, n_clusters=10,
                 mergeCluster=True, plt=False, pltNo=0):
        super().__init__(train_X, train_y, test_X, test_y, mergeCluster, plt, pltNo)
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.n_clusters = n_clusters

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
        """Cluster a single column of data using BIRCH."""
        # Extract data
        spl = (self.train_y if is_target else self.train_X)[column_name].to_numpy().reshape(-1, 1)
        test_spl = (self.test_y if is_target else self.test_X)[column_name].to_numpy().reshape(-1, 1)

        # Standardize the data
        # scaler = StandardScaler()
        # spl_scaled = scaler.fit_transform(spl)
        # test_spl_scaled = scaler.transform(test_spl)

        # Perform BIRCH clustering
        if self.threshold is None:
            self.threshold = self._find_optimal_threshold(spl)
        birch = Birch(threshold=self.threshold, 
                      branching_factor=self.branching_factor, 
                      n_clusters=self.n_clusters)
        cls1 = birch.fit_predict(spl)

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

    def _find_optimal_threshold(self, data, threshold_range=np.arange(0.01, 0.1, 0.01)):
        """Find the optimal threshold using silhouette score."""
        silhouette_scores = []
        for threshold in threshold_range:
            birch = Birch(threshold=threshold, 
                          branching_factor=self.branching_factor, 
                          n_clusters=self.n_clusters)
            labels = birch.fit_predict(data)
            score = silhouette_score(data, labels)
            silhouette_scores.append(score)

        optimal_threshold = threshold_range[np.argmax(silhouette_scores)]
        return optimal_threshold

    def plot_threshold_selection(self, data, column_name, threshold_range=np.arange(0.1, 1.1, 0.1)):
        """Plot silhouette scores for different threshold values."""
        spl = data[column_name].to_numpy().reshape(-1, 1)
        # scaler = StandardScaler()
        # spl_scaled = scaler.fit_transform(spl)

        silhouette_scores = []
        for threshold in threshold_range:
            birch = Birch(threshold=threshold, 
                          branching_factor=self.branching_factor, 
                          n_clusters=self.n_clusters)
            labels = birch.fit_predict(spl)
            score = silhouette_score(spl, labels)
            silhouette_scores.append(score)

        plt.figure(figsize=(10, 5))
        plt.plot(threshold_range, silhouette_scores, marker='o')
        plt.title(f'Silhouette Score vs. Threshold for {column_name}')
        plt.xlabel('Threshold')
        plt.ylabel('Silhouette Score')
        plt.show()