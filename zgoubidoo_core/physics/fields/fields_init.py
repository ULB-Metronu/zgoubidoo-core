from numba import jit
import numpy as np


@jit(nopython=True)
def init_field() -> tuple:
    field_partials = (np.zeros(3, dtype=np.float64),  # B
                      np.zeros((3, 3), dtype=np.float64),  # DB in original Zgoubi
                      np.zeros((3, 3, 3), dtype=np.float64),  # DDB
                      np.zeros((3, 3, 3, 3), dtype=np.float64),  # D3B
                      np.zeros((3, 3, 3, 3, 3), dtype=np.float64)  # D4B
                      )
    return field_partials


@jit(nopython=True)
def b_partials_unif_z(r: np.ndarray, val: float = 1.) -> tuple:
    """
    Uniform B field along the Z axis
    """
    b_partials = init_field()
    b_partials[0][2] = val  # Bz = 1
    return b_partials


@jit(nopython=True)
def quad_field(r: np.ndarray, b0, r0, xl):
    g0 = b0/r0
    b_partials = init_field()
    if  0 <= r[0] <= xl :
        b_partials[0][1] = g0*r[2]  # by = G0*z
        b_partials[0][2] = g0*r[1]  # bz = G0*y

        b_partials[1][1, :] = np.array((0, 0, g0))  # db/dy = (0,0,g0)
        b_partials[1][2, :] =  np.array((0, g0, 0))  # db/dz = (0,g0,0)
        return b_partials
    else:
        return b_partials
