"""Zgoubidoo's integration method.

    Todo : complete doc
"""
import numpy as np

from physics.coordinates import Coordinates
from physics.field import Field
import numpy as _np


def find_b(x, y, z) -> Field:
    """
    Todo : really find B from surrounding objects
    :param x:
    :param y:
    :param z:
    :return: The magnetic field of a drift in the Z direction
    """
    b_x = 0.0
    b_y = 0.0
    b_z = 1.0

    return Field(b_x, b_y, b_z)


def find_e(x, y, z) -> Field :
    """
    Todo : really find E from surrounding objects
    :param x:
    :param y:
    :param z:
    :return: The identically zero magnetic field
    """
    e_x = 0.0
    e_y = 0.0
    e_z = 1.0

    return Field(e_x, e_y, e_z)


def iteration(coords: Coordinates, rigidity: float, step: float):
    """An iteration of the ray-tracking process

    :param coords:
    :return:
    """
    (x, y, z) = coords.cartesian()
    u = coords.u()

    b = find_b(x, y, z)
    e = find_e(x, y, z)

    # Compute the derivatives of the speed to compute Taylors dev
    if not np.any(b) and not np.any(e):
        # Fields are all zero
        # TODO : derive u depending on the fields
        print("No fields")

    elif not np.any(b):
        print("B is 0")
    elif not np.any(e):
        print("E is 0")
    else:
        print("E and B are not 0")
