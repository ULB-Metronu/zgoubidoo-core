from zgoubidoo_core.physics.coordinates import Coordinates
from zgoubidoo_core.physics.particles import Particle
from zgoubidoo_core.physics.fields.fields_init import b_partials_unif_z, init_field
import zgoubidoo_core.tracker as tracker

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

    res = tracker.integrate(p, b1, e, max_step, step)
    df = results_to_df(res)
    plot_pos(position_from_res(df))


if __name__ == '__main__':
    main()
