from zgoubidoo_core.physics.coordinates import Coordinates
from zgoubidoo_core.physics.particles import Particle
from zgoubidoo_core.physics.fields import fields_init
from zgoubidoo_core import tracker
import numpy as np
from numba import jit
import pytest


def test_integrate():
    p = Particle(Coordinates(0, 0, 0, 0, 0, 1), 1)
    # tracker.integrate(p, (0, 0, 0), (0, 0, 1), 10, .01)
    print("todo : integrate")
    assert 1+1 == 2


def test_detect_field():
    e = fields_init.init_field()
    b_partials = fields_init.b_partials_unif_z()
    assert not np.any(e[0])
    assert np.any(b_partials[0])


def test_taylor_const_velocity():
    u_derivs = np.zeros((6, 3))
    u_derivs[0, 2] = 1  # Constant velocity
    r_m0 = np.zeros(3)
    step = .001
    new_r, new_u = tracker.taylors(r_m0, u_derivs, step)
    real_r1 = np.zeros(3)
    real_r1[2] = .001
    assert (new_r == real_r1).all
    assert (new_u == u_derivs[0, :]).all


def test_taylor():
    u_derivs = np.zeros((6, 3))
    u_derivs[0, 2] = 1  # Constant velocity
    u_derivs[1, 0] = .5  # du/ds_x = 1/2, velocity evolving along x axis
    r_m0 = np.zeros(3)
    step = 10e-3
    new_r, new_u = tracker.taylors(r_m0, u_derivs, step)

    real_r1 = np.zeros(3)
    real_r1[0] = 2.5e-07
    real_r1[2] = .001

    real_u = u_derivs[0, :]
    real_u[0] = .5*step

    assert (new_u == real_u).all
    assert (new_r == real_r1).all

