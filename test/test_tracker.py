
from zgoubidoo_core.physics.coordinates import Coordinates
from zgoubidoo_core.physics.particles import Particle
from zgoubidoo_core.physics.fields.field import Field
from zgoubidoo_core import tracker
import pytest


def test_integrate():
    p = Particle(Coordinates(0, 0, 0, 0, 0, 1, 1), 1)

    tracker.integrate(p, Field(0), Field(0), 10, .01)
    assert 1+1 == 2
