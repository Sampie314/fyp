import numpy as np
import pandas as pd

from sklearn.neighbors import KernelDensity
from matplotlib.pyplot import plot
from scipy.signal import argrelextrema
import torch
import torch.nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import math
from sklearn.metrics import r2_score
from torch.nn import TransformerEncoder, TransformerEncoderLayer




def padData(X, X_test, padDim):
    zeros_array = np.zeros((X.shape[0], padDim))
    zeros_array_test = np.zeros((X_test.shape[0], padDim))

    X = np.concatenate((X, zeros_array), axis=1)
    X_test = np.concatenate((X_test, zeros_array_test), axis=1)
    return X, X_test

def testData(pred, Y_test):
    res = (pred-Y_test)**2
    res[np.isnan(res)]=0
    mse = np.mean(res)
    return mse  

def add_log_return_to_close(close_0: np.array, log_returns: np.array) -> np.array:
    """use log returns to calculate next period close prices close_1"""
    close_1 = close_0 * np.exp(log_returns)
    return close_1

def calculate_win_rate(target_returns: np.array, pred_returns: np.array) -> float:
    target_returns = target_returns > 0
    pred_returns = pred_returns > 0
    return (target_returns == pred_returns).sum() / len(pred_returns)

def custom_r2_score(y_true, y_pred):
    """
    Calculate a custom R² score where the denominator is the sum of squared
    excess values without demeaning.

    Meant to be used for scoring log returns predictions.

    Taken from: 
    Gu, Shihao and Kelly, Bryan T. and Xiu, Dacheng, Empirical Asset Pricing via Machine Learning (September 13, 2019). Chicago Booth Research Paper No. 18-04, 31st Australasian Finance and Banking Conference 2018, Yale ICF Working Paper No. 2018-09, Available at SSRN: https://ssrn.com/abstract=3159577 or http://dx.doi.org/10.2139/ssrn.3159577

    Parameters:
    y_true (array-like): True values
    y_pred (array-like): Predicted values

    Returns:
    float: Custom R² score
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate residuals
    residuals = y_true - y_pred

    # Calculate the numerator (sum of squared residuals)
    ss_res = np.sum(residuals**2)

    # Calculate the denominator (sum of squared excess values without demeaning)
    ss_tot = np.sum(y_true**2)

    # Calculate and return the custom R² score
    custom_r2 = 1 - (ss_res / ss_tot)

    return custom_r2