"""Zgoubidoo's integration method.

    Todo : complete doc
"""
import numpy as np
import scipy.special
from physics.coordinates import Coordinates
from physics.particles import Particle
from physics.fields.field_point import Field_point
from physics.fields.field import Field


def integrate(part: Particle, b: Field, e: Field, max_step: int, step_size: float):
    b_val = Field_point(0, 0, 1)
    e_val = Field_point(0, 0, 0)

    print("e : ", e_val)
    print("b : ", b_val)
    new_r = part.coords
    new_u = part.u()
    new_rigid = part.rigidity
    for i in range(max_step):
        x, y, z = part.coords.cartesian()
        new_r, new_u, new_rigidity = iteration(part.coords, new_u, new_rigidity, step_size, b_val, e_val)
        part.set_coords(new_r, new_u, new_rigidity)
        print(new_r)


def iteration(coords: Coordinates, u: np.array, rigidity: float, step: float, b: Field_point, e: Field_point):
    """An iteration of the ray-tracking process

    :param coords: The coordinates of the particle
    :param rigidity: Rigidity of the particle
    :param step: The step of the discrete integration process
    :param b: Magnetic field on the point of process
    :param e: Electric field on the point of process
    :return:
    """
    (x, y, z) = coords.cartesian()
    #u = coords.u()
    u_derivs = derive_u(b, e, rigidity, u)

    rigidity_m1 = update_rigidity(u, rigidity, e) if np.any(e.array) else rigidity

    r_m1, u_m1 = taylors(coords, u_derivs, step)
    return r_m1, u_m1, rigidity_m1


def derive_u(b: Field_point, e: Field_point, rigidity: float, u: np.array) -> np.array:
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
    elif not np.any(e):
        return derive_u_in_b(u, b, rigidity)
    elif not np.any(b):
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


def derive_u_in_b(u: np.array, b: Field_point, rigidity: float) -> np.array:
    b_derivs = np.zeros(6, 3)
    b_derivs[0, :] = b.array/rigidity
    u_derivs = np.zeros((6, 3))
    u_derivs[0, :] = u

    for i in range(5):
        for k in range(i+1):
            u_derivs[i+1, :] += scipy.special.comb(i, k) * np.cross(u_derivs[k, :], b_derivs[i-k, :])
            # TODO : add derivations of B which depend on u_derivs and partial derivs of b
    return u_derivs


def derive_u_in_e(u, e, rigidity) -> np.array:
    pass


def derive_u_in_both(u, b, e, rigidity) -> np.array:
    pass


def update_rigidity(u, rigidity, e) -> float:
    # TODO
    pass


def taylors(coords, u_derivs, step) -> (np.array, np.array):
    r_m1 = np.zeros((1, 3))
    u_m1 = np.zeros((1, 3))

    r_m1[1, :] += coords.cartesian()
    for i in range(1, 6):
        r_m1[1, :] += u_derivs[i, :]*(step**i)
        u_m1[1, :] += u_derivs[i-1, :]*(step**i)
    return r_m1, u_m1
