import plotly.graph_objects as go
import pandas as pd

def create_actions_df(env) -> pd.DataFrame:
    actions_df = pd.DataFrame(env._actions_memory)
    actions_df.columns = ['cash'] + list(env._tic_list)
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

def plot_rewards(df:pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    # Add traces for each constituent
    fig.add_trace(
        go.Scatter(x=df.index, y=df['rewards'], name='rewards', mode='lines')
    )

    # Update layout
    fig.update_layout(
        title='Rewards',
        xaxis_title='Date',
        yaxis_title='Reward',
        legend_title='Rewards',
        hovermode='x unified'
    )

    return fig

def plot_actions(df:pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    # Add traces for each constituent
    for column in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df[column], name=column, mode='lines')
        )

    # Update layout
    fig.update_layout(
        title='Portfolio Allocation',
        xaxis_title='Date',
        yaxis_title='Weight',
        legend_title='Constituents',
        hovermode='x unified'
    )

    return fig

