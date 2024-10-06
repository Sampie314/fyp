import pandas as pd
import numpy as np

def calculate_macd(df, short_window=12, long_window=26, signal_window=9) -> pd.DataFrame:
    """
    Calculate MACD (Moving Average Convergence Divergence) for a given DataFrame of close prices.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the stock's close prices with a column 'Close'.
    short_window (int): The short-term period for the fast EMA (default is 12).
    long_window (int): The long-term period for the slow EMA (default is 26).
    signal_window (int): The period for the signal line EMA (default is 9).

    Returns:
    pd.DataFrame: Original DataFrame with additional columns for MACD and Signal line.
    """
    df = df.copy()
    # Calculate the short-term EMA
    df['EMA_Fast'] = df['Close'].ewm(span=short_window, adjust=False).mean()
    
    # Calculate the long-term EMA
    df['EMA_Slow'] = df['Close'].ewm(span=long_window, adjust=False).mean()
    
    # Calculate the MACD line
    df['MACD'] = df['EMA_Fast'] - df['EMA_Slow']
    
    # Calculate the Signal line
    df['MACD_Signal_Line'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()
    
    return df[['MACD', 'MACD_Signal_Line']]

def calculate_rsi(df, period=14) -> pd.DataFrame:
    """
    Calculate the Relative Strength Index (RSI) for a given DataFrame of close prices.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the stock's close prices with a column 'Close'.
    period (int): The period over which to calculate the RSI (default is 14).

    Returns:
    pd.DataFrame: Original DataFrame with an additional column for RSI.
    """
    df = df.copy()
    
    # Calculate the price differences
    delta = df['Close'].diff()
    
    # Separate gains and losses
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    # Calculate the Relative Strength (RS)
    rs = gain / loss
    
    # Calculate RSI
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df[['RSI']]

def calculate_cci(df, period=20):
    """
    Calculate the Commodity Channel Index (CCI) for a given DataFrame of close prices.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the stock's close prices with a column 'Close'.
    period (int): The period over which to calculate the CCI (default is 20).

    Returns:
    pd.DataFrame: Original DataFrame with an additional column for CCI.
    """
    df = df.copy()
    
    # Calculate the Typical Price
    df['Typical_Price'] = (df['Close'] + df['High'] + df['Low']) / 3
    
    # Calculate the Simple Moving Average of the Typical Price
    df['SMA_Typical'] = df['Typical_Price'].rolling(window=period).mean()
    
    # Calculate the Mean Deviation
    df['Mean_Deviation'] = df['Typical_Price'].rolling(window=period).apply(lambda x: pd.Series(x).mad(), raw=True)
    
    # Calculate CCI
    df['CCI'] = (df['Typical_Price'] - df['SMA_Typical']) / (0.015 * df['Mean_Deviation'])
    
    return df[['CCI']]
