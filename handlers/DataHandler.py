import yfinance as yf
import openbb as obb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class DataHandler:
    @staticmethod
    def getData(ticker_symbol, startTrain, endTrain, startTest, endTest, num_horizons):
        data_dir = "./data/main"
        os.makedirs(data_dir, exist_ok=True)
        
        train_file_path = f"{data_dir}/{ticker_symbol}_train_preprocessed_{num_horizons}.pkl"
        test_file_path = f"{data_dir}/{ticker_symbol}_test_preprocessed_{num_horizons}.pkl"
        
        if os.path.exists(train_file_path) and os.path.exists(test_file_path):
            train_data = pd.read_pickle(train_file_path)
            test_data = pd.read_pickle(test_file_path)
            train_features, train_targets = train_data['features'], train_data['targets']
            test_features, test_targets = test_data['features'], test_data['targets']
        else:
            train_data = yf.download(ticker_symbol, start=startTrain, end=endTrain, auto_adjust=True)
            test_data = yf.download(ticker_symbol, start=startTest, end=endTest, auto_adjust=True)

            train_features, train_targets = DataHandler._preprocess(train_data, num_horizons)
            test_features, test_targets = DataHandler._preprocess(test_data, num_horizons)
        
            pd.to_pickle({'features': train_features, 'targets': train_targets}, train_file_path)
            pd.to_pickle({'features': test_features, 'targets': test_targets}, test_file_path)
        
        return train_features, train_targets, test_features, test_targets

    @staticmethod
    def _preprocess(df, num_horizons):
        y = pd.DataFrame()

        for i in range(num_horizons):
            y[f"y_log_return_{i}"] = DataHandler.log_return(df, days=i+1, delay=-i)
        
        for i in range(num_horizons):
            df[f'daily_log_return_{i}'] = DataHandler.daily_log_return(df, i)
        
        for i in range(num_horizons):
            df[f'intraday_log_return_{i}'] = DataHandler.intraday_log_return(df, i)
        
        for i in range(num_horizons):
            df[f'overnight_log_return_{i}'] = DataHandler.over_night_return(df, i)

        for i in range(num_horizons):
            df[f'day_range_{i}'] = DataHandler.day_range(df, i)

        df['notional_traded'] = DataHandler.notional_traded(df)

        for i in range(num_horizons):
            df[f'notional_traded_change_{i}'] = DataHandler.notional_traded_change(df, 1, i)

        df['num_up_days_1m'] = DataHandler.up_days_count(df, 21)

        df['1m_mom'] = DataHandler.momentum(df, 21)
        df['3m_mom'] = DataHandler.momentum(df, 63)
        df['6m_mom'] = DataHandler.momentum(df, 126)
        df['12m_mom'] = DataHandler.momentum(df, 252)
        df['18m_mom'] = DataHandler.momentum(df, 378)
        df['mom_change_1m_3m'] = DataHandler.momentum_change(df, 21, 63)
        df['mom_change_3m_6m'] = DataHandler.momentum_change(df, 63, 126)
        df['mom_change_6m_12m'] = DataHandler.momentum_change(df, 126, 252)

        # Handle NaN values in both df and y
        df.dropna(inplace=True)
        y.dropna(inplace=True)

        # Align the indices of df and y
        common_index = df.index.intersection(y.index)
        df = df.loc[common_index]
        y = y.loc[common_index]

        return df, y

    @staticmethod
    def daily_log_return(df, delay):
        return DataHandler.log_return(df, 1, delay)
    
    @staticmethod
    def log_return(df, days, delay=0):
        return np.log(df['Close'].shift(delay) / df['Close'].shift(delay + days))
    
    @staticmethod
    def over_night_return(df, delay=0):
        return np.log(df['Open'].shift(delay) / df['Close'].shift(delay + 1))
    
    @staticmethod
    def day_range(df, delay=0):
        return np.log(df['High'].shift(delay) / df['Low'].shift(delay))

    @staticmethod
    def momentum(df, days):
        return DataHandler.log_return(df, days, delay=0)

    @staticmethod
    def intraday_log_return(df, delay=0):
        return np.log(df['Close'].shift(delay) / df['Open'].shift(delay))

    @staticmethod
    def momentum_change(df, period1, period2):
        momentum1 = DataHandler.momentum(df, period1)
        momentum2 = DataHandler.momentum(df, period2)
        return momentum1 - momentum2

    @staticmethod
    def up_days_count(df, days):
        return df['Close'].diff().rolling(window=days).apply(lambda x: (x > 0).sum())

    @staticmethod
    def notional_traded(df):
        return df['Close'] * df['Volume']

    @staticmethod
    def notional_traded_change(df, days, delay=0):
        notional = DataHandler.notional_traded(df)
        return notional.pct_change(days).shift(delay)