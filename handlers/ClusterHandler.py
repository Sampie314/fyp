import numpy as np
import pandas as pd
import math
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt


class ClusterHandler:
    def __init__(self, train_X, train_y, test_X, test_y, eps=0.0005, mergeCluster=True, pltKDE=False, pltNo=0, splitLargest=True):
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
        self.eps = eps
        self.mergeCluster = mergeCluster
        self.pltKDE = pltKDE
        self.pltNo = pltNo
        self.splitLargest = splitLargest

        # Initialize dictionaries to store cluster statistics
        self.feature_cluster_stats = {}
        self.y_cluster_stats = {}

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
            # temp.reset_index(drop=True, inplace=True)
            temp.index = self.train_y.index
            testTemp.index = self.test_y.index
        else:
            # testTemp.reset_index(drop=True, inplace=True)
            testTemp.index = self.test_X.index
            temp.index = self.train_X.index

        # Append results
        # toTrain.append(temp)
        # toTest.append(testTemp)

        # Store cluster statistics
        if is_target:
            self.y_cluster_stats[column_name] = clsDic
        else:
            self.feature_cluster_stats[column_name] = clsDic
        return temp, testTemp

    def _cluster_column(self, column_name, is_target=False):
        """Cluster a single column of data."""
        # Extract data
        spl = (self.train_y if is_target else self.train_X)[column_name].to_numpy().reshape(-1, 1)
        test_spl = (self.test_y if is_target else self.test_X)[column_name].to_numpy()

        # Perform KDE and find splits
        splits = self._kde_splits(spl)

        # Assign initial clusters
        cls1 = np.sum(spl.reshape(-1)[:, np.newaxis] >= splits, axis=1)

        # Create temporary DataFrames
        temp = pd.DataFrame({'Data': spl.reshape(-1), f'Cluster_{column_name}': cls1})
        testTemp = pd.DataFrame({'Data': test_spl})

        # Calculate initial cluster statistics
        self._calculate_cluster_stats(temp, f'Cluster_{column_name}')

        # Merge small clusters if enabled
        if self.mergeCluster:
            self._merge_clusters(temp, f'Cluster_{column_name}')

        # Calculate membership functions
        clsDic = self._calculate_memberships(temp, testTemp, f'Cluster_{column_name}', column_name, is_target)

        print(f"{column_name} had {len(clsDic)} clusters")
        return temp, testTemp, clsDic

    def _kde_splits(self, spl):
        """Perform KDE and find splits in the data."""
        kde = KernelDensity(kernel='gaussian', bandwidth=self.eps).fit(spl)
        s = np.linspace(min(spl), max(spl))
        e = kde.score_samples(s.reshape(-1, 1))
        mi = argrelextrema(e, np.less)[0]
        splits = s[mi].reshape(-1)

        if self.splitLargest:
            splits = self._split_largest_cluster(splits)

        return splits

    def _split_largest_cluster(self, splits):
        """Split the largest cluster."""
        differences = np.abs(np.diff(splits))
        max_diff_index = np.argmax(differences)
        middle_element = (splits[max_diff_index] + splits[max_diff_index + 1]) / 2
        return np.insert(splits, max_diff_index + 1, middle_element)

    def _calculate_cluster_stats(self, df, cluster_col):
        """Calculate mean and std for each cluster."""
        for cluster_id, cluster_data in df.groupby(cluster_col):
            df.loc[df[cluster_col] == cluster_id, 'Cluster_Mean'] = cluster_data['Data'].mean()
            df.loc[df[cluster_col] == cluster_id, 'Cluster_Std'] = cluster_data['Data'].std(ddof=1)

    def _merge_clusters(self, df, cluster_col):
        """Merge small clusters."""
        clusterNo = sorted(df[cluster_col].unique())
        i, j = 0, 1
        while i < len(clusterNo) and j < len(clusterNo):
            lCls, rCls = clusterNo[i], clusterNo[j]
            if len(df[df[cluster_col] == lCls]) < 5 or len(df[df[cluster_col] == rCls]) < 5:
                df.loc[df[cluster_col] == rCls, cluster_col] = lCls
                newMean = df[df[cluster_col] == lCls]['Data'].mean()
                newStd = df[df[cluster_col] == lCls]['Data'].std()
                df.loc[df[cluster_col] == lCls, 'Cluster_Mean'] = newMean
                df.loc[df[cluster_col] == lCls, 'Cluster_Std'] = newStd
                clusterNo.pop(j)
            else:
                i += 1
                j += 1

    def _calculate_memberships(self, temp, testTemp, cluster_col, column_name, is_target):
        """Calculate membership functions for each cluster."""
        clsDic = {}
        for clsNo, cluster_data in enumerate(temp.groupby(cluster_col)):
            _, data = cluster_data
            sampleMean = data['Data'].mean()
            sampleDevi = data['Data'].std(ddof=1)

            # Handle prefix assignment
            prefix = column_name
            pdf_col = f'PDF_{prefix}_{clsNo}'

            temp[pdf_col] = temp["Data"].apply(lambda x: self._membership(x, sampleMean, sampleDevi))
            testTemp[pdf_col] = testTemp["Data"].apply(lambda x: self._membership(x, sampleMean, sampleDevi))

            clsDic[clsNo] = {'mean': sampleMean, 'std': sampleDevi}

        return clsDic

    def _membership(self, x, sampleMean, sampleDevi):
        """Calculate Gaussian membership."""
        return math.exp(-((x - sampleMean)**2 / (2 * (sampleDevi**2))))

    def clusterGraph(self):
        """Generate cluster distribution graphs."""
        return
        fig, axs = plt.subplots(12, 3, figsize=(45,55))  # 1 row, 2 columns
        for z in range(0, 17): # 16 time steps, ignore t
            a = np.where(self.means[z] == 0)[0]
            if len(a)==0:
                a = len(self.means[z])
            else:
                a = a[0]
            for i in range(a):
                axs[z//3][z%3].scatter(self.train["Close_t-{}".format(z)], self.train["PDF_{}_{}".format(z, i)])
            axs[z//3][z%3].set_xlabel('Closing Value at t-{}'.format(z), fontsize = 20)
            axs[z//3][z%3].set_ylabel('Membership Value', fontsize = 20)
            axs[z//3][z%3].set_title('Cluster Distribution for t-{}'.format(z), fontsize = 30)
            
        for z in range(0, 16): # 15 time steps, ignore t
            a = np.where(self.meansMomentum[z] == 0)[0]
            if len(a)==0:
                a = len(self.meansMomentum[z])
            else:
                a = a[0]
            for i in range(a):
                axs[(z+16)//3][(z+16)%3].scatter(self.train["Close_t-{}.2".format(z)], self.train["PDF_{}.2_{}".format(z, i)])
            axs[(z+16)//3][(z+16)%3].set_xlabel('Closing Value at t-{}.2'.format(z), fontsize = 20)
            axs[(z+16)//3][(z+16)%3].set_ylabel('Membership Value', fontsize = 20)
            axs[(z+16)//3][(z+16)%3].set_title('Cluster Distribution for t-{}.2'.format(z), fontsize = 30)

        for i in range(len(self.y_cluster_stats)):
                axs[-1][-1].scatter(self.train["Close_t+{}".format(self.yTarget)], self.train["PDF_y_{}".format(i)])
        axs[-1][-1].set_xlabel('Closing Value at t+{}'.format(self.yTarget), fontsize = 20)
        axs[-1][-1].set_ylabel('Membership Value', fontsize = 20)
        axs[-1][-1].set_title('Cluster Distribution for t-{}'.format(self.yTarget), fontsize = 30)
        plt.tight_layout()    
        plt.show()

    def deFuzzify(self, pred: np.array, target_col: str) -> np.array:
        """
        defuzzification uising weighted average technique
        transform fuzzy memberships -> log returns
        """
        centers = [dic['mean'] for dic in self.y_cluster_stats[target_col].values()]
        stds = [dic['std'] for dic in self.y_cluster_stats[target_col].values()]
    
        centers = np.tile(centers, (pred.shape[0], 1))
        pred = np.nan_to_num(pred, nan=0, posinf=0, neginf=0)
    
        denominator = pred.sum(axis=1, keepdims=True)
        numerator = (pred * centers).sum(axis = 1, keepdims=True)
        result = numerator / denominator
        result = np.squeeze(result)
    
        return result