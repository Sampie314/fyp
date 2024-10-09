import pandas as pd
import numpy as np
import math
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import time

from numpy import array, linspace
from sklearn.neighbors import KernelDensity
from matplotlib.pyplot import plot
from scipy.signal import argrelextrema
import random

import torch
import torch.nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import math
from sklearn.metrics import r2_score
from torch.nn import TransformerEncoder, TransformerEncoderLayer

    
from handlers.DataHandler import DataHandler
from handlers.AgglomerativeHandler import AgglomerativeHandler as ch
from handlers import GAHandler
from handlers import Utils
from handlers import Model

### GLOBAL VARIABLES ###
y_horizon = 13

top100 = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'GOOG',
       'LLY', 'JPM', 'AVGO', 'TSLA', 'UNH', 'XOM', 'V', 'PG', 'JNJ', 'MA',
       'COST', 'HD', 'ABBV', 'WMT', 'MRK', 'NFLX', 'KO', 'BAC', 'ADBE',
       'PEP', 'CVX', 'CRM', 'TMO', 'ORCL', 'LIN', 'AMD', 'ACN', 'MCD',
       'ABT', 'CSCO', 'PM', 'WFC', 'IBM', 'TXN', 'QCOM', 'GE', 'DHR',
       'VZ', 'INTU', 'AMGN', 'NOW', 'ISRG', 'NEE', 'SPGI', 'PFE', 'CAT',
       'DIS', 'RTX', 'CMCSA', 'GS', 'UNP', 'T', 'AMAT', 'PGR',
       'LOW', 'AXP', 'TJX', 'HON', 'BKNG', 'ELV', 'COP', 'SYK', 'MS',
       'LMT', 'VRTX', 'BLK', 'REGN', 'MDT', 'BSX', 'PLD', 'CB', 'ETN',
       'C', 'MMC', 'ADP', 'AMT', 'PANW', 'ADI', 'SBUX', 'MDLZ', 'CI',
       'TMUS', 'FI', 'BMY', 'DE', 'GILD', 'BX', 'NKE', 'SO', 'LRCX', 'MU', 'KLAC', 'SCHW']

top10 = top100[:10]

best_chromosome = {'num_heads': 1, 'feed_forward_dim': 32, 'num_transformer_blocks': 2, 
                   'mlp_units': 64, 'dropout_rate': 0.1, 'learning_rate': 0.0001, 
                   'num_mlp_layers': 8, 'num_epochs': 50, 'activation_function': 0, 'batch_size': 1024}

train_X, train_y, test_X, test_y = DataHandler.getTickers(top10, "1998-01-01", "2015-12-31", "2016-01-01", "2024-01-01", y_horizon)

# Create and use the Cluster object
stationary_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'notional_traded']

cls = ch(train_X.drop(columns=stationary_cols + ['symbol']), train_y.drop(columns=['symbol']), 
                     test_X.drop(columns=stationary_cols + ['symbol']), test_y.drop(columns=['symbol']), 
                     n_clusters=None, max_clusters=10, mergeCluster=True)

clustered_train_X, clustered_train_y, clustered_test_X, clustered_test_y = cls.cluster()

X, t_X = clustered_train_X.to_numpy(), clustered_test_X.to_numpy()

close_preds = {}
log_returns_preds = {}

for y_period in range(13):
    pred_col = f'y_log_return_{y_period}'
    print(f'------ Y target: {pred_col} ------')
    
    y = clustered_train_y[[c for c in clustered_train_y.columns if pred_col + "_" in c]].to_numpy()
    t_y = clustered_test_y[[c for c in clustered_test_y.columns if pred_col + "_" in c]].to_numpy()
    crisp_train_y = train_y[pred_col].to_numpy()
    crisp_t_y = test_y[pred_col].to_numpy()
    
    final_model, final_r2, pred = Model.train_model(
        X, y, t_X, t_y, crisp_t_y, cls, pred_col,
        **best_chromosome
    )
    print("Final model R2 score:", final_r2)
    pred_log_returns = cls.deFuzzify(pred, pred_col)
    pred_log_returns = np.nan_to_num(pred_log_returns, nan=0, posinf=0, neginf=0)
    
    pred_close = Utils.add_log_return_to_close(test_X['Close'].to_numpy(), pred_log_returns)
    target_close = Utils.add_log_return_to_close(test_X['Close'].to_numpy(), test_y[pred_col].to_numpy())
    close_r2_score = r2_score(target_close, pred_close)
    win_rate = Utils.calculate_win_rate(crisp_t_y, pred_log_returns)

    close_preds[pred_col] = pred_close
    log_returns_preds[pred_col] = pred_log_returns
    
    print(f"Stock Price R2: {close_r2_score:.4f}")
    print(f"Win Rate: {100 * win_rate:.4f}%")

close_preds = pd.DataFrame(close_preds, index=test_X.index)
close_preds.columns = ['close+1', 'close+2', 'close+3', 'close+4', 'close+5', 
                       'close+6', 'close+7', 'close+8', 'close+9', 'close+10', 
                       'close+11', 'close+12', 'close+13']
close_preds['symbol'] = test_X['symbol']
close_preds.to_pickle('predictions/agglomerative_close_preds.pkl')

log_returns_preds = pd.DataFrame(log_returns_preds, index=test_X.index)
log_returns_preds['symbol'] = test_X['symbol']
log_returns_preds.to_pickle('predictions/agglomerative_log_returns_preds.pkl')