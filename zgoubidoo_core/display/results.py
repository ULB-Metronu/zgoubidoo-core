import plotly.graph_objects as go
import numpy as np


def plot(res):
    positions = np.zeros((3, len(res)+1))
    for idx, val in enumerate(res):
        positions[:, idx] += val[0]

    fig = go.Figure(data=go.Scatter3d(
        x=positions[0, :], y=positions[1, :], z=positions[2, :],
        marker=dict(
            size=4,
            color=positions[0,:],
            colorscale='Viridis',
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
