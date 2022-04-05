import os

from zgoubidoo_core.physics.coordinates import Coordinates
from zgoubidoo_core.physics.particles import Particle
from zgoubidoo_core.physics.fields.fields_init import b_partials_unif_z, init_field
import zgoubidoo_core.tracker as tracker
import numpy as np
import math

from zgoubidoo_core.postprocessing.display.results import *
from zgoubidoo_core.postprocessing.data.results import results_to_df

from numba import jit


@jit(nopython=True)
def b1(r: np.array) -> tuple:
    res = b_partials_unif_z(r, 2.3)
    return res


@jit(nopython=True)
def b2(r: np.ndarray) -> tuple:
    partials = init_field()
    partials[0][2] = 0.5*r[2] + 2.3  # Bz = 2.3 + 0.5z
    partials[1][2, 2] = 0.5   # dB/dz = (0, 0, 0.5)
    return partials

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


def main():
    """
    Main function of Zgoobidoo-core, is to be deleted
    :return: None
    """
    print("Welcome in zgoubidoo_core")
    c = Coordinates(0, 0, 0, 0, 0, 1)
    p = Particle(coords=c, rigidity=2.32182)  # Proton with 230MeV
    max_step = 1000
    step = 10e-4

    res = tracker.integrate(p, bend_1m, e, max_step, step)
    df = results_to_df(res)
    compare_res_csv(df, 'Data/bend_validation/data_nominal.csv')


def test_compute_correspondance():
    max_step = 999
    step = 10e-4

    c_nominal = Coordinates(0, 0, 0, 0, 0, 1)
    c_offset_y = Coordinates(0.01, 0, 0, 0, -0.00108, 1)
    c_offset_yp = Coordinates(0, 0.001, 0, 0, 0, 1)
    c_offset_z = Coordinates(0, 0, 0.01, 0.000233, 0, 1)
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


def test_diffs_correspondance():
    dfs = test_compute_correspondance()
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
        zgoubis_dfs.append(pandas.read_csv('Data/bend_validation/' + f))

    dist = lambda row: math.sqrt(math.pow(row.X - row.zgoubX, 2) +
                                 math.pow(row.Y - row.zgoubY, 2) +
                                 math.pow(row.Z - row.zgoubZ, 2))
    for idx, df in enumerate(dfs):
        df['zgoubX'] = zgoubis_dfs[idx]['X']
        df['zgoubY'] = zgoubis_dfs[idx]['Y']
        df['zgoubZ'] = zgoubis_dfs[idx]['Z']
        df['dist'] = df.apply(dist, axis=1)
        print(df.head())
        fig = px.line(df, x=np.arange(0, 1000, 1), y='dist', title=filenames[idx])
        fig.show()




def test_plot_correspondance():
    dfs = test_compute_correspondance()

    compare_res_csv(dfs[0], 'Data/bend_validation/data_nominal.csv')
    compare_res_csv(dfs[1], 'Data/bend_validation/data_offset_y.csv')
    compare_res_csv(dfs[2], 'Data/bend_validation/data_offset_yp.csv')
    compare_res_csv(dfs[3], 'Data/bend_validation/data_offset_z.csv')
    compare_res_csv(dfs[4], 'Data/bend_validation/data_offset_zp.csv')
    compare_res_csv(dfs[5], 'Data/bend_validation/data_offset_dr.csv')


if __name__ == '__main__':
    #main()
    # test_diffs_correspondance()
    test_plot_correspondance()
