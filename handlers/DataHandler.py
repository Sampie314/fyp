import yfinance as yf
import openbb as obb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import sys

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

class DataHandler:

    @staticmethod
    def getTickers(symbols:list[str], startTrain: str, endTrain: str, startTest: str, endTest: str, num_horizons: int):
        """Get the data for the given symbols, combine data into a single dataframe and return the train and test data."""
        train_features = []
        train_targets = []
        test_features = []
        test_targets = []

        for symbol in symbols:
            train_f, train_t, test_f, test_t = DataHandler.getData(symbol, startTrain, endTrain, startTest, endTest, num_horizons)
            train_f['symbol'] = symbol
            train_t['symbol'] = symbol
            test_f['symbol'] = symbol
            test_t['symbol'] = symbol
            train_features.append(train_f)
            train_targets.append(train_t)
            test_features.append(test_f)
            test_targets.append(test_t)

        train_features = pd.concat(train_features)
        train_targets = pd.concat(train_targets)
        test_features = pd.concat(test_features)
        test_targets = pd.concat(test_targets)

        train_features.sort_index(inplace=True)
        train_targets.sort_index(inplace=True)
        test_features.sort_index(inplace=True)
        test_targets.sort_index(inplace=True)

        return train_features, train_targets, test_features, test_targets

    @staticmethod
    def getData(ticker_symbol, startTrain, endTrain, startTest, endTest, num_horizons):
        data_dir = "./data/main"
        os.makedirs(data_dir, exist_ok=True)
        
        file_path = f"{data_dir}/{ticker_symbol}_{startTrain}_{endTest}.pkl"

        if os.path.exists(file_path):
            logger.info(f"Loading data for {ticker_symbol} from cache")
            data = pd.read_pickle(file_path)
        else:
            logger.info(f"Downloading data for {ticker_symbol}")
            data = yf.download(ticker_symbol, start=startTrain, end=endTest, auto_adjust=True)
            if data.empty:
                raise ValueError(f"No data found for {ticker_symbol}")
            
            data.to_pickle(file_path)

        features, targets = DataHandler._preprocess(data, num_horizons)

        # train_features.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
        # test_features.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)

        features = DataHandler._replace_inf_nan_with_rolling_mean(features)

        # Handle NaN values in TRAIN
        features.dropna(inplace=True)
        targets.dropna(inplace=True)

        # Align the indices of df and y
        common_index = features.index.intersection(targets.index)
        features = features.loc[common_index]
        targets = targets.loc[common_index]

        # LOOK FORWARD CHECK
        DataHandler._lookforward_bias_check(features, targets)
        
        train_features, train_targets = features.loc[startTrain:endTrain], targets.loc[startTrain:endTrain]
        test_features, test_targets = features.loc[startTest:endTest], targets.loc[startTest:endTest]

        

        return train_features, train_targets, test_features, test_targets

    @staticmethod
    def _replace_inf_nan_with_rolling_mean(df, window=10):
        df_cleaned = df.copy()
        numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            rolling_mean = df_cleaned[col].rolling(window=window, min_periods=1).mean()
            invalid_mask = np.isinf(df_cleaned[col]) | np.isnan(df_cleaned[col])

            df_cleaned.loc[invalid_mask, col] = rolling_mean[invalid_mask]
        
        return df_cleaned

    @staticmethod
    def _preprocess(df, num_horizons):
        y = pd.DataFrame()

        for i in range(num_horizons):
            y[f"y_log_return_{i}"] = DataHandler.target_log_return(df, i+1)
        
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
        df['returns_volatility_1m'] = DataHandler.returns_std(df, 21)
        df['returns_volatility_3m'] = DataHandler.returns_std(df, 63)
        df['close_cv_1m'] = DataHandler.coefficient_of_variation(df, 21)
        df['close_cv_3m'] = DataHandler.coefficient_of_variation(df, 63)
        

        # Handle NaN values in both df and y
        df.dropna(inplace=True)
        y.dropna(inplace=True)

        # Align the indices of df and y
        common_index = df.index.intersection(y.index)
        df = df.loc[common_index]
        y = y.loc[common_index]

        return df, y

    @staticmethod
    def target_log_return(df, t):
        return np.log(df['Close'].shift(-t) / df['Close'])
    
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

    @staticmethod
    def coefficient_of_variation(df, days):
        return  df['Close'].rolling(days).std() / df['Close'].rolling(days).mean()

    @staticmethod
    def returns_std(df, days):
        return DataHandler.daily_log_return(df, 0).rolling(days).std()

    @staticmethod
    def _lookforward_bias_check(feature_df, target_df):
        returns_cols1= ['daily_log_return_0',
       'daily_log_return_1', 'daily_log_return_2', 'daily_log_return_3',
       'daily_log_return_4', 'daily_log_return_5', 'daily_log_return_6',
       'daily_log_return_7', 'daily_log_return_8', 'daily_log_return_9',
       'daily_log_return_10', 'daily_log_return_11', 'daily_log_return_12']
    
        t_cols = ['y_log_return_0', 'y_log_return_1', 'y_log_return_2', 'y_log_return_3',
               'y_log_return_4', 'y_log_return_5', 'y_log_return_6', 'y_log_return_7',
               'y_log_return_8', 'y_log_return_9', 'y_log_return_10',
               'y_log_return_11', 'y_log_return_12']

        for r1 in returns_cols1:
            for r2 in t_cols:
                if (feature_df[r1] == target_df[r2]).all():
                    raise Exception(f"Feature ColumnL: {r1} matches Target Column: {r2}!")