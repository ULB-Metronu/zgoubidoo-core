from typing import List, Tuple

import pandas
import plotly.graph_objects as go
import plotly.express as px
import numpy as np


def plot_pos(positions):
    fig = go.Figure(data=go.Scatter3d(
        x=positions['X'],
        y=positions['Y'],
        z=positions['Z'],
        marker=dict(
            size=1,
        ),
        line=dict(
            color='darkblue',
            width=2
        )
    ))

    fig.update_layout(
        width=800,
        height=700,
        autosize=False,
        scene=dict(
            camera=dict(
                up=dict(
                    x=0,
                    y=0,
                    z=1
                ),
                eye=dict(
                    x=0,
                    y=1.0707,
                    z=1,
                )
            ),
            aspectratio=dict(x=1, y=1, z=0.7),
            aspectmode='manual'
        ),
    )
    fig.show()


def position_from_res(res: pandas.DataFrame) -> pandas.DataFrame:
    return res[['X', 'Y', 'Z']]


def plot_csv_xy(filename):
    df = pandas.read_csv(filename)
    fig = px.line(df, x='X', y='Y', title=filename)
    fig.show()


def plot_both_trajectories(df: pandas.DataFrame, title: str, col_names: List[Tuple[str, str]]):
    fig = go.Figure(
        layout=go.Layout(
            title=go.layout.Title(text=title)
        )
    )

    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=df[col_names[0][0]], y=df[col_names[0][1]],
            marker=dict(
                size=10,
                color='Red',
            ),
            name='Zgoubidoo',
        )
    )

    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=df[col_names[1][0]], y=df[col_names[1][1]],
            marker=dict(
                size=5,
                color='LightBlue',
                symbol='cross'
            ),
            name='Zgoubi',
        )
    )
    fig.show()
