import math
import os

import numpy as np
import pandas
from numba import jit

from zgoubidoo_core import tracker
from zgoubidoo_core.physics.coordinates import Coordinates
from zgoubidoo_core.physics.fields.fields_init import b_partials_unif_z, init_field
from zgoubidoo_core.physics.particles import Particle
from zgoubidoo_core.postprocessing.data.results import results_to_df
from zgoubidoo_core.postprocessing.display.results import compare_res_csv

import plotly.express as px


@jit(nopython=True)
def bend_1m(r: np.ndarray) -> tuple:
    if 0 <= r[0] <= 1:
        return b_partials_unif_z(r, 0.5)
    else:
        return init_field()


@jit(nopython=True)
def e(r: np.array):
    res = b_partials_unif_z(r, 0)  # E(r) = 0
    return res


def compute_correspondance():
    max_step = 999
    step = 10e-4

    c_nominal = Coordinates(0, 0, 0, 0, 0, 1)
    c_offset_y = Coordinates(0.01, 0, 0, 0, -0.0010830381, 1)
    c_offset_yp = Coordinates(0, 0.001, 0, 0, 0, 1)
    c_offset_z = Coordinates(0, 0, 0.01, 0.00023323042777606976, 0, 1)
    c_offset_zp = Coordinates(0, 0, 0, 0.001, 0, 1)
    coords = [
        c_nominal,
        c_offset_y,
        c_offset_yp,
        c_offset_z,
        c_offset_zp
    ]
    parts = []
    for c in coords:
        parts.append(Particle(coords=c, rigidity=2.32182))
    parts.append(Particle(coords=c_nominal, rigidity=2.554002))

    dfs = []
    for p in parts:
        res = tracker.integrate(p, bend_1m, e, max_step, step)
        dfs.append(results_to_df(res))
    return dfs


def test_diffs():
    dfs = compute_correspondance()
    filenames = [
        'data_nominal.csv',
        'data_offset_y.csv',
        'data_offset_yp.csv',
        'data_offset_z.csv',
        'data_offset_zp.csv',
        'data_offset_dr.csv',
    ]
    zgoubis_dfs = []
    for f in filenames:
        if os.path.isfile('Data/bend_validation/' + f):
            zgoubis_dfs.append(pandas.read_csv('Data/bend_validation/' + f))
        else:
            print('Missing file "' + f, '"exiting.')
            print(os.getcwd())
            exit()

    xdist = lambda row: math.sqrt(math.pow(row.X - row.zgoubX, 2))
    ydist = lambda row: math.sqrt(math.pow(row.Y - row.zgoubY, 2))
    zdist = lambda row: math.sqrt(math.pow(row.Z - row.zgoubZ, 2))

    dist = lambda row: math.sqrt(math.pow(row.X - row.zgoubX, 2) +
                                 math.pow(row.Y - row.zgoubY, 2) +
                                 math.pow(row.Z - row.zgoubZ, 2))
    for idx, df in enumerate(dfs):
        df['zgoubX'] = zgoubis_dfs[idx]['X']
        df['zgoubY'] = zgoubis_dfs[idx]['Y']
        df['zgoubZ'] = zgoubis_dfs[idx]['Z']
        df['dist'] = df.apply(dist, axis=1)
        fig = px.line(df, x=np.arange(0, 1000, 1), y='dist', title=filenames[idx])
        fig.show()


def test_plot_correspondence():
    dfs = compute_correspondance()

    compare_res_csv(dfs[0], 'Data/bend_validation/data_nominal.csv')
    compare_res_csv(dfs[1], 'Data/bend_validation/data_offset_y.csv')
    compare_res_csv(dfs[2], 'Data/bend_validation/data_offset_yp.csv')
    compare_res_csv(dfs[3], 'Data/bend_validation/data_offset_z.csv')
    compare_res_csv(dfs[4], 'Data/bend_validation/data_offset_zp.csv')
    compare_res_csv(dfs[5], 'Data/bend_validation/data_offset_dr.csv')


def main():
    if os.path.isdir('Data'):
        test_diffs()
        test_plot_correspondence()
    else:
        print('No Data folder')


if __name__ == '__main__':
    main()
