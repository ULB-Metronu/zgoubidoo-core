from zgoubidoo_core.physics.coordinates import Coordinates
from zgoubidoo_core.physics.particles import Particle
import zgoubidoo_core.tracker as tracker
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from numba import jit


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


@jit(nopython=True)
def b(r: np.array) -> np.array:
    res = np.zeros((6, 3))
    res[0, 2] = 1  # B_z = 1
    return res


@jit(nopython=True)
def e(r: np.array):
    res = np.zeros((6, 3))
    return res


def main():
    """
    Main function of Zgoobidoo-core, is to be deleted
    :return: None
    """
    print("Welcome in zgoubidoo_core")
    c = Coordinates()
    print("coords :", c.cartesian())
    p = Particle(coords=c, rigidity=1)
    max_step = 400
    step = 10e-4
    res = tracker.integrate(p, b, e, max_step, step)

    plot(res)


if __name__ == '__main__':
    main()
