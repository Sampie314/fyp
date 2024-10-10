import pandas as pd

def reindex(df: pd.DataFrame, date_col: str = 'Date', ticker_col:str = 'symbol') -> pd.DataFrame:
    """reindex and sort the dataframe to prepare for RLenv"""
    df = df.copy()
    df = df.sort_values([ticker_col, date_col], ignore_index=True)
    df.index = df[date_col].factorize()[0]
    return df

def create_actions_df(env, cash=True) -> pd.DataFrame:
    actions_df = pd.DataFrame(env._actions_memory)
    if cash:
        actions_df.columns = ['cash'] + list(env._tic_list)
    else:
        actions_df.columns = list(env._tic_list)
        
    actions_df.index = env._date_memory
    return actions_df

def create_metrics_df(env) -> pd.DataFrame:
    metrics_df = pd.DataFrame(
                {
                    "date": env._date_memory,
                    "returns": env._portfolio_return_memory,
                    "rewards": env._portfolio_reward_memory,
                    "portfolio_values": env._asset_memory["final"],
                }
                )
    return metrics_df.set_index('date')