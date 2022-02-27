from numba import jit
import numpy as np


@jit(nopython=True)
def init_field() -> tuple:
    field_partials = (np.zeros(3),  # B
                      np.zeros((3, 3)),  # DB in original Zgoubi
                      np.zeros((3, 3, 3)),  # DDB
                      np.zeros((3, 3, 3, 3)),  # D3B
                      np.zeros((3, 3, 3, 3, 3))  # D4B
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
