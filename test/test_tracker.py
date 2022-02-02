from zgoubidoo_core.physics.coordinates import Coordinates
from zgoubidoo_core.physics.particles import Particle
from zgoubidoo_core import tracker
import pytest


def test_integrate():
    p = Particle(Coordinates(0, 0, 0, 0, 0, 1, 1), 1)

    tracker.integrate(p, (0, 0, 0), (0, 0, 1), 10, .01)
    assert 1+1 == 2
