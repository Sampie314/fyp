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
# from handlers.KDEHandler import KDEHandler as ch
from handlers.KMeansHandler import KMeansHandler as ch
from handlers import GAHandler
from handlers import Utils

### GLOBAL VARIABLES ###
y_horizon = 13

top100 = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'BRK', 'GOOG',
       'LLY', 'JPM', 'AVGO', 'TSLA', 'UNH', 'XOM', 'V', 'PG', 'JNJ', 'MA',
       'COST', 'HD', 'ABBV', 'WMT', 'MRK', 'NFLX', 'KO', 'BAC', 'ADBE',
       'PEP', 'CVX', 'CRM', 'TMO', 'ORCL', 'LIN', 'AMD', 'ACN', 'MCD',
       'ABT', 'CSCO', 'PM', 'WFC', 'IBM', 'TXN', 'QCOM', 'GE', 'DHR',
       'VZ', 'INTU', 'AMGN', 'NOW', 'ISRG', 'NEE', 'SPGI', 'PFE', 'CAT',
       'DIS', 'RTX', 'CMCSA', 'GS', 'UNP', 'T', 'AMAT', 'PGR',
       'LOW', 'AXP', 'TJX', 'HON', 'BKNG', 'ELV', 'COP', 'SYK', 'MS',
       'LMT', 'VRTX', 'BLK', 'REGN', 'MDT', 'BSX', 'PLD', 'CB', 'ETN',
       'C', 'MMC', 'ADP', 'AMT', 'PANW', 'ADI', 'SBUX', 'MDLZ', 'CI',
       'TMUS', 'FI', 'BMY', 'DE', 'GILD', 'BX', 'NKE', 'SO', 'LRCX', 'MU', 'KLAC']

top10 = top100[:10]

# train_X, train_y, test_X, test_y = DataHandler.getTickers(top100, "1998-01-01", "2015-12-31", "2016-01-01", "2024-01-01", y_horizon)
train_X, train_y, test_X, test_y = DataHandler.getTickers(top10, "1998-01-01", "2015-12-31", "2016-01-01", "2024-01-01", y_horizon)
# train_X, train_y, test_X, test_y = DataHandler.getData("MS", "1998-01-01", "2015-12-31", "2016-01-01", "2023-01-01", y_horizon)

train_X.drop(columns=['symbol'], inplace=True)
train_y.drop(columns=['symbol'], inplace=True)
test_X.drop(columns=['symbol'], inplace=True)
test_y.drop(columns=['symbol'], inplace=True)

stationary_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'notional_traded']
# train_X.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
# test_X.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)

# Create and use the Cluster object
cls = ch(train_X.drop(columns=stationary_cols), train_y, 
                     test_X.drop(columns=stationary_cols), test_y, 
                     max_clusters=3, mergeCluster=True)
clustered_train_X, clustered_train_y, clustered_test_X, clustered_test_y = cls.cluster()

class TransformerModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=10, ff_dim=32, num_transformer_blocks=4, mlp_units=256, dropout=0.25, noHiddenLayers=1, sigmoidOrSoftmax=0):
        super(TransformerModel, self).__init__()

        # print(f"Initializing TransformerModel with input_dim: {input_dim}, output_dim: {output_dim}")

        # # Ensure input_dim is divisible by num_heads
        # if input_dim % num_heads != 0:
        #     new_input_dim = (input_dim // num_heads + 1) * num_heads
        #     print(f"Adjusting input_dim from {input_dim} to {new_input_dim} to be divisible by num_heads ({num_heads})")
        #     input_dim = new_input_dim

        # Encoder layer with ff_dim and dropout
        encoder_layers = TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_transformer_blocks)
        
        # Dropout layer
        self.dropout = torch.nn.Dropout(dropout)
        
        # MLP layers
        fcList = [torch.nn.Linear(input_dim, mlp_units), torch.nn.ReLU(), self.dropout]
        for i in range(noHiddenLayers):
            fcList.extend([
                torch.nn.Linear(mlp_units, mlp_units//2),
                torch.nn.ReLU(),
                self.dropout
            ])
            mlp_units = mlp_units//2
        fcList.append(torch.nn.Linear(mlp_units, output_dim))
        
        # Output activation
        if sigmoidOrSoftmax == 0:
            fcList.append(torch.nn.Sigmoid())
        else:
            fcList.append(torch.nn.Softmax(dim=-1))
        
        self.fc = torch.nn.Sequential(*fcList)

    def forward(self, src):
        # src shape: (batch_size, seq_length, input_dim)
        
        # Pass input through the transformer encoder
        encoder_output = self.transformer_encoder(src)
        
        # Apply dropout after the transformer encoder
        encoder_output = self.dropout(encoder_output)
        
        # Pass through the MLP layers
        output = self.fc(encoder_output)
        return output
    
def train_model(X, Y, X_test, Y_test, # fuzzified inputs
                   Y_test_raw, # crisp value
                   num_heads, feed_forward_dim, num_transformer_blocks, mlp_units, dropout_rate, 
                   learning_rate, num_mlp_layers, num_epochs, activation_function, batch_size):

    OUTPUT_FREQ = 50
    # Detect GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Load and pad data
    x_padded, x_test_padded = Utils.padData(X, X_test, math.ceil(X.shape[1] / num_heads) * num_heads - X.shape[1])

    # Initialize the model
    model = TransformerModel(
        input_dim=x_padded.shape[1], 
        output_dim=Y.shape[1],
        num_heads=num_heads, 
        ff_dim=feed_forward_dim, 
        num_transformer_blocks=num_transformer_blocks, 
        mlp_units=mlp_units, 
        dropout=dropout_rate, 
        sigmoidOrSoftmax=activation_function
    ).double()

    # Send model to the detected device
    model = model.to(device)

    # Set loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create DataLoader for training
    assert x_padded.shape[0] == Y.shape[0]
    train_dataset = TensorDataset(torch.from_numpy(x_padded), torch.from_numpy(Y))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0
        model.train()

        all_preds = []
        all_targets = []

        for inputs, targets in train_dataloader:
            # Move data to the correct device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            epoch_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Collect predictions and targets for R² score computation
            all_preds.append(outputs.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())

        # Concatenate all batch predictions and targets
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        if epoch % OUTPUT_FREQ == 0:
            # Compute R² score
            r2 = r2_score(all_targets, all_preds)
            log_return_r2 = eval_model(model, x_test_padded, Y_test, Y_test_raw)
            
            print(f'Epoch [{epoch+1}/{num_epochs}] | Loss: {epoch_loss / len(train_dataloader):.4f} | Train Cluster R² Score: {r2:.4f} | Test Log Return R² Score: {log_return_r2:.4f}')

    # return all_preds, all_targets

    # Evaluate on test data
    test_dataset = TensorDataset(torch.from_numpy(x_test_padded), torch.from_numpy(Y_test))
    test_dataloader = DataLoader(test_dataset, batch_size=512*4, shuffle=False)

    model.eval()
    pred_list = []
    with torch.no_grad():
        for inputs, _ in test_dataloader:
            pred = model(inputs.to(device)).cpu()
            pred_list.append(pred)

    mse = Utils.testData(np.concatenate(pred_list), Y_test)
    # print(f'Test Data MSE: {mse:.7f}')

    # Calculate R2 score
    pred = np.concatenate(pred_list)
    pred[np.isnan(pred)] = 0

    pred_log_returns = cls.deFuzzify(pred, pred_col)
    pred_log_returns = np.nan_to_num(pred_log_returns, nan=0, posinf=0, neginf=0)
    
    # pred_closing = addResToClosing(cls, res)[:-yTarget]
    # actual_closing = cls.test.Close[yTarget:].to_numpy()
    # r2_score_value = r2_score(actual_closing, pred_closing)
    
    r2_score_value = r2_score(Y_test_raw, pred_log_returns)
    print("Test R2 Score:", r2_score_value)

    return model, r2_score_value, pred

def eval_model(model, X: np.array, Y: np.array, Y_crisp: np.array):
    """
    X & Y are fuzzified inputs
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using {device}")    

    model = model.to(device)
    model.eval()
    pred_list = []

    test_dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    test_dataloader = DataLoader(test_dataset, batch_size=512*4, shuffle=False)
    
    with torch.no_grad():
        for inputs, _ in test_dataloader:
            pred = model(inputs.to(device)).cpu()
            pred_list.append(pred)

    # mse = Utils.testData(np.concatenate(pred_list), Y_test)
    # print(f'Test Data MSE: {mse:.7f}')

    # Calculate R2 score
    pred = np.concatenate(pred_list)
    pred[np.isnan(pred)] = 0

    pred_log_returns = cls.deFuzzify(pred, pred_col)
    pred_log_returns = np.nan_to_num(pred_log_returns, nan=0, posinf=0, neginf=0)
    
    r2_score_value = Utils.custom_r2_score(Y_crisp, pred_log_returns)
    # print("Test R2 Score:", r2_score_value)    
    return r2_score_value


pred_col = test_y.columns[0]

train_val_split = 0.7
unique_train_dates = clustered_train_X.index.unique()
split_date = unique_train_dates[int(len(unique_train_dates) * train_val_split)]


X = clustered_train_X.loc[clustered_train_X.index < split_date].to_numpy()
val_X = clustered_train_X.loc[clustered_train_X.index >= split_date].to_numpy()

y = clustered_train_y[[c for c in clustered_train_y.columns if pred_col in c]].loc[clustered_train_y.index < split_date].to_numpy()
val_y = clustered_train_y[[c for c in clustered_train_y.columns if pred_col in c]].loc[clustered_train_y.index >= split_date].to_numpy()
crisp_val_y = train_y[pred_col].loc[train_y.index >= split_date].to_numpy()

# Start the timer
start_time = time.time()

# Run the genetic algorithm
best_chromosome, best_fitness = GAHandler.genetic_algorithm(
    train_model, X, y, val_X, val_y, crisp_val_y, 
    initial_population_size=50, final_population_size=5, generations=10, elite_size=2
)

# End the timer
end_time = time.time()

# Calculate the execution time
execution_time = end_time - start_time

# Print the results
print("Best hyperparameters:", best_chromosome)
print("Best R2 score:", best_fitness)
print(f"Execution time: {execution_time:.4f} seconds")
print(f"Execution time: {execution_time/60:.4f} minutes")
