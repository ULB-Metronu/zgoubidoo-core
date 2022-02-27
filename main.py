from zgoubidoo_core.physics.coordinates import Coordinates
from zgoubidoo_core.physics.particles import Particle
import zgoubidoo_core.tracker as tracker
import numpy as np
from zgoubidoo_core.display.results import plot

from numba import jit


@jit(nopython=True)
def b(r: np.array) -> np.array:
    res = np.zeros((6, 3))
    res[0, 2] = 2.3  # B_z = 1
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
    p = Particle(coords=c, rigidity=2.19)  # Proton with 230MeV
    max_step = 1000
    step = 10e-3
    res = tracker.integrate(p, b, e, max_step, step)

    plot(res)


if __name__ == '__main__':
    main()
