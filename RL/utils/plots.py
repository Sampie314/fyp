import plotly.graph_objects as go
import pandas as pd

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
        hovermode='x unified',
        
        width=1400,   # Set the width of the figure
        height=800,  # Set the height of the figure
        # plot_bgcolor='white',  # Set plot background to white
        # xaxis=dict(showline=True, linewidth=2, linecolor='black'),
        # yaxis=dict(showline=True, linewidth=2, linecolor='black')
        
    )

    return fig

