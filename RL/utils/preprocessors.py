import pandas as pd

def reindex(df: pd.DataFrame, date_col: str = 'Date', ticker_col:str = 'symbol') -> pd.DataFrame:
    """reindex and sort the dataframe to prepare for RLenv"""
    df = df.copy()
    df = df.sort_values([ticker_col, date_col], ignore_index=True)
    df.index = df[date_col].factorize()[0]
    return df
