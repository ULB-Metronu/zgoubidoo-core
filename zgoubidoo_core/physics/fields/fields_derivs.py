import numpy as np
from numba import jit


@jit(nopython=True)
def derive_field(field_partials, u_derivs, order: int) -> np.ndarray:
    """
    Compute the order'th derivatives of a field (noted "b") over a displacement ds : dB/ds


    Field partials are the partial derivatives of b at current point s.t.
    b[i][j, k, l, ...] = B^(j, k, l...)_i where i is the ith cartesian coordinate
    :param field_partials: Partial derivatives of the field
    :param u_derivs: Derivatives du/ds up to order i-1
    :param order: Order of derivative to compute
    :return: Derivatives of a field up to a given order
    """
    if order < 0 or order > 4:
        print("Error : wrong derivation order :", order)
        print("considering order 0")
        return field_partials[0]

    if order == 0:
        return d0_field(field_partials, u_derivs)
    elif order == 1:
        return d1_field(field_partials, u_derivs)
    elif order == 2:
        return d2_field(field_partials, u_derivs)
    elif order == 3:
        return d3_field(field_partials, u_derivs)
    elif order == 4:
        return d4_field(field_partials, u_derivs)

    print("Programming error at zgoubidoo_core.physics.fields.fields_derivs.py")
    return field_partials[0]  # Should not reach this


@jit(nopython=True)
def d0_field(field_partials, u_derivs) -> np.ndarray:
    return field_partials[0]


@jit(nopython=True)
def d1_field(field_partials, u_derivs) -> np.ndarray:
    res = np.zeros(3)
    for i in range(3):
        res += field_partials[1][:, i]*u_derivs[0, i]
    return res


@jit(nopython=True)
def d2_field(field_partials, u_derivs) -> np.ndarray:
    res = np.zeros(3)
    b_parts1 = field_partials[1]
    b_parts2 = field_partials[2]
    for i in range(3):
        res += b_parts1[:, i]*u_derivs[1, i]
        for j in range(3):
            res += b_parts2[:, i, j]*u_derivs[0, i]*u_derivs[0, j]
    return res


@jit(nopython=True)
def d3_field(field_partials, u_derivs) -> np.ndarray:
    res = np.zeros(3)
    temp = np.zeros(3)
    u = u_derivs[0, :]
    b_parts1 = field_partials[1]
    b_parts2 = field_partials[2]
    b_parts3 = field_partials[3]

    for i in range(3):
        res += b_parts1[:, i]*u_derivs[2, i]
        for j in range(3):
            temp += b_parts2[:, i, j]*u_derivs[1, i]*u[j]
            for k in range(3):
                res += b_parts3[:, i, j, k]*u[i]*u[j]*u[k]
    return res + (3*temp)


@jit(nopython=True)
def d4_field(field_partials, u_derivs) -> np.ndarray:
    res = np.zeros(3)
    temp3 = np.zeros(3)
    temp4 = np.zeros(3)
    temp6 = np.zeros(3)
    u = u_derivs[0, :]
    b_parts1 = field_partials[1]
    b_parts2 = field_partials[2]
    b_parts3 = field_partials[3]
    b_parts4 = field_partials[4]

    for i in range(3):
        res += b_parts1[:, i]*u_derivs[3, i]
        for j in range(3):
            temp3 += b_parts2[:, i, j]*u_derivs[1, i]*u_derivs[1, j]
            temp4 += b_parts2[:, i, j]*u_derivs[2, i]*u[j]
            for k in range(3):
                temp6 += b_parts3[:, i, j, k]*u_derivs[1, i]*u[j]*u[k]
                for l in range(3):
                    res += b_parts4[:, i, j, k, l]*u[i]*u[j]*u[k]*u[l]

    return res + (3*temp3) + (4*temp4) + (6*temp6)

