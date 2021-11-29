"""Zgoubidoo's integration method.

    Todo : complete doc
"""
import numpy as np
import scipy.special

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
    :param x: Longitudinal coordinate
    :param y: Horizontal plane coordinate
    :param z: Vertical plane coordinate
    :return: The identically zero electric field
    """
    e_x = 0.0
    e_y = 0.0
    e_z = 0.0

    return Field(e_x, e_y, e_z)


def update_rigidity(u, rigidity, e) -> float:
    pass


def iteration(coords: Coordinates, rigidity: float, step: float, b: Field, e: Field):
    """An iteration of the ray-tracking process

    :param coords: The coordinates of the particule
    :param rigidity: Rigidity of the particle
    :param step: The step of the discrete integration process
    :return:
    """
    (x, y, z) = coords.cartesian()
    u = coords.u()
    u_derivs = derive_u(b, e, rigidity, u)

    if np.any(e.array):
        # E is not 0
        new_rigidity = update_rigidity(u, rigidity, e)


def derive_u(b: Field, e: Field, rigidity: float, u: np.array) -> np.array:
    """
    Computes the derivatives of u at the current point M0. Uses different methods according to the presence of fields.
    :param b: Magnetic field at M0
    :param e: Electric field at M0
    :param rigidity: Rigidity of the particle at M0
    :param u: Unitary velocity vector of the particle at M0
    :return: The derivatives of u to the sixth order
    """
    if not np.any(b) and not np.any(e):
        return derive_u_no_fields(u)
    elif not np.any(b):
        return derive_u_in_b(u, b, rigidity)
    elif not np.any(e):
        return derive_u_in_e(u, e, rigidity)
    else:
        return derive_u_in_both(u, b, e, rigidity)


def derive_u_no_fields(u: np.array) -> np.array:
    """
    Compute the derivatives of u when no field is present. Thus u remains constant
    :param u: Unitary velocity vector
    :return: A five by three array. Line i represents the ith derivative of u (d^i)u/(ds)^i
    """
    u_derivs = np.zeros((5, 3))
    u_derivs[0, :] = u
    return u_derivs


def derive_b(b: Field, u: np.array) -> np.array:
    """
    TODO derive B correctly
    :param b:
    :param u:
    :return:
    """
    b_derivs = np.zeros((6,3))
    print("b array", b.array)
    b_derivs[0, :] = b.array

    print(b_derivs)
    #b_derivs[1, :] = np.sum()
    return b_derivs


def derive_u_in_b(u: np.array, b: Field, rigidity: float) -> np.array:
    b_derivs = derive_b(b, u)
    u_derivs = np.zeros((6, 3))
    u_derivs[0, :] = u

    print("u_derivs", u_derivs)

    for i in range(5):
        for k in range(i+1):
            add = scipy.special.comb(i, k) * np.cross(u_derivs[k, :], b_derivs[i-k, :])
            print("add", add)
            u_derivs[i+1, :] += add
    return u_derivs


def derive_u_in_e(u, e, rigidity) -> np.array:
    pass


def derive_u_in_both(u, b, e, rigidity) -> np.array:
    pass

