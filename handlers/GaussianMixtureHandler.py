import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
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

class GaussianMixtureHandler(AbstractClusterHandler):
    """Gaussian Mixture Model clustering handler."""
    def __init__(self, train_X, train_y, test_X, test_y, 
                covariance_type='full', max_iter=100, n_init=1,
                 mergeCluster=True, plt=False, pltNo=0):
        super().__init__(train_X, train_y, test_X, test_y, mergeCluster, plt, pltNo)
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.n_init = n_init

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
        """Cluster a single column of data using Gaussian Mixture Model."""
        # Extract data
        spl = (self.train_y if is_target else self.train_X)[column_name].to_numpy().reshape(-1, 1)
        test_spl = (self.test_y if is_target else self.test_X)[column_name].to_numpy().reshape(-1, 1)

        # Standardize the data
        # scaler = StandardScaler()
        # spl_scaled = scaler.fit_transform(spl)
        # test_spl_scaled = scaler.transform(test_spl)

        # Determine the number of components if not specified
        n_components = self._find_optimal_components(spl)

        # Perform Gaussian Mixture clustering
        gmm = GaussianMixture(n_components=n_components, 
                              covariance_type=self.covariance_type,
                              max_iter=self.max_iter,
                              n_init=self.n_init,
                              random_state=42)
        cls1 = gmm.fit_predict(spl)

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

    def _find_optimal_components(self, data, max_components=10):
        """Find the optimal number of components using BIC."""
        n_components_range = range(1, min(max_components + 1, len(data)))
        bic = []
        for n_components in n_components_range:
            gmm = GaussianMixture(n_components=n_components, 
                                  covariance_type=self.covariance_type,
                                  max_iter=self.max_iter,
                                  n_init=self.n_init,
                                  random_state=42)
            gmm.fit(data)
            bic.append(gmm.bic(data))
        
        return n_components_range[np.argmin(bic)]

    def plot_bic(self, data, column_name, max_components=10):
        """Plot the BIC scores for different numbers of components."""
        spl = data[column_name].to_numpy().reshape(-1, 1)
        scaler = StandardScaler()
        spl_scaled = scaler.fit_transform(spl)

        n_components_range = range(1, min(max_components + 1, len(spl_scaled)))
        bic = []
        for n_components in n_components_range:
            gmm = GaussianMixture(n_components=n_components, 
                                  covariance_type=self.covariance_type,
                                  max_iter=self.max_iter,
                                  n_init=self.n_init,
                                  random_state=42)
            gmm.fit(spl_scaled)
            bic.append(gmm.bic(spl_scaled))

        plt.figure(figsize=(10, 5))
        plt.plot(n_components_range, bic, marker='o')
        plt.title(f'BIC Score vs. Number of GMM Components for {column_name}')
        plt.xlabel('Number of Components')
        plt.ylabel('BIC Score')
        plt.show()