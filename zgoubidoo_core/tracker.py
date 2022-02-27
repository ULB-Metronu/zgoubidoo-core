"""Zgoubidoo's integration method.

    Todo : complete doc
"""
import math
from types import FunctionType

from numba import jit
import numpy as np
import scipy.special

from zgoubidoo_core.physics.coordinates import Coordinates
from zgoubidoo_core.physics.particles import Particle


def integrate(part: Particle, b: '(r: ndarray) -> tuple', e: '(r: ndarray) -> tuple', max_step: int, step_size: float):
    r = part.cartesian()  # np_array x,y,z
    u = part.u()
    new_rigid = part.rigidity
    results = integr_loop(b, e, max_step, r, new_rigid, u, step_size)
    return results


# @jit(nopython=True)
def integr_loop(b: '(r: ndarray) -> ndarray',
                e: '(r: ndarray) -> ndarray',
                max_step: int,
                new_r: np.ndarray,
                new_rigid: float,
                new_u: np.ndarray,
                step_size: float):
    results = [(np.copy(new_r), np.copy(new_u), np.copy(new_rigid))]
    for i in range(max_step):
        new_r, new_u, new_rigid = iteration(new_r, new_u, new_rigid, step_size, b, e)
        # print('new_r :', new_r)
        results.append((np.copy(new_r), np.copy(new_u), np.copy(new_rigid)))
        # print()
    return results


@jit(nopython=True)
def iteration(r: np.array, u: np.array, rigidity: float, step: float, b: FunctionType, e: FunctionType):
    """An iteration of the ray-tracking process

    The functions b and e must return the partial derivatives of B and E s.t.
    b(r)[i, j] = B^(i)_j
    e(r)[i, j] = E^(i)_j

    :param r: Coordinates of the particle
    :param u: Unit velocity of the particle
    :param rigidity: Rigidity of the particle
    :param step: The step of the discrete integration process
    :param b: Magnetic field on the point of process
    :param e: Electric field on the point of process
    :return:
    """
    b_partials: np.array = b(r)
    e_partials: np.array = e(r)
    u_derivs = derive_u(b_partials, e_partials, rigidity, u)

    if np.any(e_partials[0, :]):
        rigidity = update_rigidity(u, rigidity, e)

    r_m1, u_m1 = taylors(r, u_derivs, step)
    return r_m1, u_m1, rigidity


@jit(nopython=True)
def derive_u(b_partials: np.array, e_partials: np.array, rigidity: float, u: np.array) -> np.array:
    """
    Computes the derivatives of u at the current point M0. Uses different methods according to the presence of fields.
    :param b_partials: Magnetic field at M0
    :param e_partials: Electric field at M0
    :param rigidity: Rigidity of the particle at M0
    :param u: Unitary velocity vector of the particle at M0
    :return: The derivatives of u to the sixth order
    """
    B = b_partials[0, :]
    E = e_partials[0, :]
    if not np.any(B) and not np.any(E):
        return derive_u_no_fields(u)
    elif not np.any(E):
        return derive_u_in_b(u, b_partials, rigidity)
    elif not np.any(B):
        return derive_u_in_e(u, e_partials, rigidity)
    else:
        return derive_u_in_both(u, b_partials, e_partials, rigidity)


@jit(nopython=True)
def derive_u_no_fields(u: np.array) -> np.array:
    """
    Compute the derivatives of u when no field is present. Thus u remains constant
    :param u: Unitary velocity vector
    :return: A 5x3 array. Line i represents the ith derivative of u (d^i)u/(ds)^i
    """
    u_derivs = np.zeros((5, 3))
    u_derivs[0, :] = u
    return u_derivs


@jit(nopython=True)
def derive_u_in_b(u: np.array, b_partials: np.array, rigidity: float) -> np.array:
    b_derivs = np.zeros((6, 3))
    b_derivs[0, :] = b_partials[0, :] / rigidity
    u_derivs = np.zeros((6, 3))
    u_derivs[0, :] = u

    for i in range(5):
        for k in range(i+1):
            u_derivs[i+1, :] += binom(i, k) * np.cross(u_derivs[k, :], b_derivs[i-k, :])
            # TODO : add derivations of B which depend on u_derivs and partial derivs of b
    # print(u_derivs)
    return u_derivs


@jit(nopython=True)
def derive_u_in_e(u, e_partials, rigidity) -> np.array:
    pass


@jit(nopython=True)
def derive_u_in_both(u, b_partials, e_partials, rigidity) -> np.array:
    pass


@jit(nopython=True)
def update_rigidity(u, rigidity, e) -> float:
    # TODO
    return rigidity


@jit(nopython=True)
def taylors(r_m0: np.array, u_derivs, step) -> (np.array, np.array):
    u_m1 = np.zeros(3)

    r_m1 = r_m0
    for i in range(6):
        r_m1 += u_derivs[i, :]*(math.pow(step, i+1))/factorial(i+1)
        u_m1 += u_derivs[i, :]*(math.pow(step, i))/factorial(i)
    return r_m1, u_m1


@jit(nopython=True)
def factorial(n):
    fact = 1
    for i in range(1, n + 1):
        fact = fact * i
    return fact


@jit(nopython=True)
def binom(n, k):
    return factorial(n) // factorial(k) // factorial(n - k)
