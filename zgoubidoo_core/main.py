from physics.coordinates import Coordinates
from physics.particles import Particle
import tracker
import pandas as pd
import plotly.graph_objects as go


def plot(res):
    fig = go.Figure(data=go.Scatter3d(
        x=res[0, :], y=res[1, :], z=res[2, :],
        marker=dict(
            size=4,
            color=res[0,:],
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


def main():
    """
    Main function of Zgoobidoo-core, is to be deleted
    :return: None
    """
    print("Welcome in zgoubidoo_core")
    c = Coordinates()
    print("coords :", c.cartesian())
    p = Particle(coords=c, rigidity=10e-5)
    b = tuple()
    e = tuple()
    max_step = 100
    step = 10e-6
    #res = tracker.integrate(p, b, e, max_step, step)
    #print(res)
    plot(res)


if __name__ == '__main__':
    main()
