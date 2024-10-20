import numpy as np
import pandas as pd
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler
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

class OPTICSHandler(AbstractClusterHandler):
    """OPTICS clustering handler."""
    def __init__(self, train_X, train_y, test_X, test_y, 
                 min_samples=0.01, max_eps=np.inf,
                 mergeCluster=True, plt=False, pltNo=0):
        super().__init__(train_X, train_y, test_X, test_y, mergeCluster, plt, pltNo)
        self.min_samples = min_samples
        self.max_eps = max_eps

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
        """Cluster a single column of data using OPTICS."""
        # Extract data
        spl = (self.train_y if is_target else self.train_X)[column_name].to_numpy().reshape(-1, 1)
        test_spl = (self.test_y if is_target else self.test_X)[column_name].to_numpy().reshape(-1, 1)

        # Perform OPTICS clustering
        clusterer = OPTICS(min_samples=self.min_samples, max_eps=self.max_eps)
        cls1 = clusterer.fit_predict(spl)

        # Handle noise points (labeled as -1 by OPTICS)
        cls1[cls1 == -1] = max(cls1) + 1

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

    def plot_reachability(self, data, column_name):
        """Plot the reachability plot for OPTICS clustering."""
        spl = data[column_name].to_numpy().reshape(-1, 1)

        clusterer = OPTICS(min_samples=self.min_samples, max_eps=self.max_eps, 
                           )
        clusterer.fit(spl)

        reachability = clusterer.reachability_
        ordering = clusterer.ordering_

        plt.figure(figsize=(10, 5))
        plt.plot(range(len(reachability)), reachability[ordering])
        plt.title(f'Reachability Plot for {column_name}')
        plt.xlabel('Points')
        plt.ylabel('Reachability Distance')
        plt.show()