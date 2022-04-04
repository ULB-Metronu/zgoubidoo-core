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
    fig = px.line(df, x='X', y='Y')
    return fig


def compare_res_csv(res: pandas.DataFrame, csv_file: str):
    df: pandas.DataFrame = pandas.read_csv(csv_file)
    res['id'] = 'zgoubidoo'
    positions = res[['X', 'Y', 'id']]
    df['id'] = 'zgoubi'
    print(df.head())
    print(positions.head())
    df = pandas.concat([df, positions])
    print(df)
    fig = px.scatter(df, x="X", y="Y", color="id")
    fig.show()
