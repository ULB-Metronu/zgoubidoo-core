from __future__ import annotations

import math
import os
from typing import List, Dict, Tuple

import numpy as np
import pandas
import pandas as pd
from numba import jit

import zgoubidoo_core.postprocessing.display.results
from zgoubidoo_core import tracker
from zgoubidoo_core.physics.coordinates import Coordinates
from zgoubidoo_core.physics.fields.fields_init import b_partials_unif_z, init_field, quad_field
from zgoubidoo_core.physics.particles import Particle
from zgoubidoo_core.postprocessing.data.results import results_to_df
from zgoubidoo_core.postprocessing.display.results import plot_both_trajectories, set_default_layout

import plotly.express as px


@jit(nopython=True)
def bend_1m(r: np.ndarray) -> Tuple:
    if 0 <= r[0] <= 1:
        return b_partials_unif_z(r, 0.5)
    else:
        return init_field()


@jit(nopython=True)
def quad_1m(r: np.ndarray) -> Tuple:
    b0 = 0.1
    r0 = 0.01
    return quad_field(r, b0=b0, r0=r0, xl=1)


@jit(nopython=True)
def e(r: np.array):
    res = init_field()  # E(r) = 0
    return res


def coords_from_zgoubi_df(df):
    x = df.iloc[0]['X']
    y = df.iloc[0]['Y']
    z = df.iloc[0]['Z']
    t = df.iloc[0]['T']
    p = df.iloc[0]['P']
    return Coordinates.from_list([x, y, z, t, p])


def rename_zgoubis_columns(df: pd.DataFrame):
    return df.rename(columns={'X': 'zgoubX', 'Y': 'zgoubY', 'Z': 'zgoubZ'})


def compute_correspondence(csv_files: dict[str: float | int],
                           b_field,
                           e_field,
                           step=10e-4,
                           data_dir="Data") \
        -> Dict[str: pd.DataFrame] | None:
    """
    Computes the correspondence between the zgoubidoo and zgoubi computations given in the csv files.
    This tracks particles according to the first coordinate set of every csv_file and the given brho.

    :param step:
    :param e_field:
    :param b_field:
    :param csv_files: Dictionnary where the keys are filenames (csv files from zgoubi computations) and values are their
    initial rigidity
    :param data_dir: Optionnal directory where the csv_files are.
    :return: A dict of pair filename / dataframes containing tracked particles coordinates or None if there was an error
    """

    dfs = {}
    for f, rigidity in csv_files.items():
        if os.path.isfile(f):
            df = pandas.read_csv(f)
            df.drop(df.tail(1).index, inplace=True)  # Remove last line as zgoubi computes special last coords
            max_step = df.shape[0]  # Number of line of csv_file (without header)
            print("Integrating", f, "for", max_step, "steps of size", step, '\n')
            c = coords_from_zgoubi_df(df)
            p = Particle(c, rigidity)
            res = tracker.integrate(p, b_field, e_field, max_step, step)
            zgoubidoo_df = results_to_df(res)
            # To avoid having columns with the same name
            df = rename_zgoubis_columns(df)
            df = df.join(zgoubidoo_df)
            dfs[f] = df
        else:
            print('Missing file "' + f + '", could not compute.')
            print(os.getcwd())
            continue

    return dfs


def plot_dists(dfs: Dict[str: pd.DataFrame], dist_method=None):
    """
    Plots the point wise distance between the elements from zgoubis and zgoubidoos integrated trajectories
    :param dfs:
    :param dist_method: Euclidian distance is used by default, available options are 'x', 'y', 'z'
    :return: None
    """

    def xdist(row):
        return abs(row.X - row.zgoubX)

    def ydist(row):
        # return math.sqrt(abs(row.Y - row.zgoubY))
        return row.Y - row.zgoubY

    def zdist(row):
        return abs(row.Z - row.zgoubZ)

    def dist(row):
        return math.sqrt(math.pow(row.X - row.zgoubX, 2) +
                         math.pow(row.Y - row.zgoubY, 2) +
                         math.pow(row.Z - row.zgoubZ, 2))

    dist_axis = "euclidian"
    if dist_method is None:
        dist_method = dist
    elif dist_method == 'x':
        dist_method = xdist
        dist_axis = "along x"
    elif dist_method == 'y':
        dist_method = ydist
        dist_axis = "along y"
    elif dist_method == 'z':
        dist_method = zdist
        dist_axis = "along z"
    else:
        dist_method = dist

    for filename, df in dfs.items():
        df['dist'] = df.apply(dist_method, axis=1, raw=False)
        fig = px.line(df, x=np.arange(0, df.shape[0], 1), y='dist', title=filename + ", distance is " + dist_axis)
        set_default_layout(fig)
        fig.show()


def plot_correspondence(dfs: Dict, x_axis='X', y_axis='Y'):
    """
    Plot the trajectories computed by zgoubi and zgoubidoo

    :param dfs: dict of the form {filename :dataframe}
    :return: None
    """
    if x_axis == 'Y':
        zgoudooX = 'Y'
        zgoubiX = 'zgoubY'
    elif x_axis == 'Z':
        zgoudooX = 'Z'
        zgoubiX = 'zgoubZ'
    else:
        print('Default value of x axis used')
        zgoudooX = 'X'
        zgoubiX = 'zgoubX'
    if y_axis == 'Y':
        zgoudooY = 'Y'
        zgoubiY = 'zgoubY'
    elif y_axis == 'Z':
        zgoudooY = 'Z'
        zgoubiY = 'zgoubZ'
    else:
        print('Default value of y axis used')
        zgoudooY = 'X'
        zgoubiY = 'zgoubX'

    for f_name, df in dfs.items():
        # TODO plot_both_trajectories(df, [(zgoudooX, zgoudooY), (zgoubiX, zgoubiY)], f_name)
        plot_both_trajectories(df, [(zgoudooX, zgoudooY), (zgoubiX, zgoubiY)])


def y_offset_distance(y_offset=0.1):
    """
    Computes the point wise distance between the trajectory of a particle starting at 0,0,0 and a particle starting at
    0, y_offset, 0

    :param y_offset:
    """
    c = Coordinates(0, 0, 0, 0, 0, 1)
    c1 = Coordinates(0, y_offset, 0, 0, 0, 1)

    p = Particle(coords=c, rigidity=2.32182)  # Proton with 230MeV
    p1 = Particle(coords=c1, rigidity=2.32182)  # Proton with 230MeV

    max_step = 1000
    step = 10e-4

    res = tracker.integrate(p, bend_1m, e, max_step, step)
    df = results_to_df(res)
    df = zgoubidoo_core.postprocessing.display.results.position_from_res(df)

    res = tracker.integrate(p1, bend_1m, e, max_step, step)
    df1 = results_to_df(res)
    df1 = zgoubidoo_core.postprocessing.display.results.position_from_res(df1)

    df1 = df1.rename(columns={'X': 'off_X', 'Y': 'off_Y', 'Z': 'off_Z'})

    df = df.join(df1)

    def ydist(row):
        # return abs(abs(row.Y - row.off_Y) - y_offset)
        return row.Y - (row.off_Y - y_offset)

    def xdist(row):
        return abs(row.X - row.off_X) - y_offset

    def dist(row):
        return math.sqrt(math.pow(row.X - row.off_X, 2) +
                         math.pow(row.Y - row.off_Y, 2) +
                         math.pow(row.Z - row.off_Z, 2)) - y_offset
    df['dist'] = df.apply(ydist, axis=1, raw=False)
    fig = px.line(df, x=np.arange(0, df.shape[0], 1), y='dist',
                  title="Normal vs Offset y of " + str(y_offset) + ", distance is along y")
    fig.show()


def main():
    dir_path = 'Data/test_dir'

    # List files from the data directory
    files: List[str] = []
    for (f, dir_names, f_names) in os.walk(dir_path):
        for f_name in f_names:
            files.append(f+os.sep+f_name)

    # Generate dict from file list
    file_dict = {}
    default_brho = 2.32182
    for filename in files:
        tokens = filename.split(sep=".")
        tok = "".join(tokens[:-1])

        # get brho %age
        p = float(tok.split(sep='_')[-1])/100
        #p = 1
        file_dict[filename] = p*default_brho

    # Calculation procedure
    dfs = compute_correspondence(file_dict,
                                 b_field=bend_1m,
                                 e_field=e)
    #plot_dists(dfs, dist_method='')
    # plot_dists(dfs, dist_method='x')
    plot_dists(dfs, dist_method='y')
    plot_correspondence(dfs, x_axis='X', y_axis='Y')


if __name__ == '__main__':
    main()

    # offsets = [0.01, 0.1, 0.5, 1]
    # for offset in offsets:
    #     y_offset_distance(offset)
