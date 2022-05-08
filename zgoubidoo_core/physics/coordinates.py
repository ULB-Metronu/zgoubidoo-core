"""This module provides a clean interface to the Zgoubi coordinates system (Y-T-Z-P-X-D).

A dataclass is used for the implementation but conversions to common formats are also provided.
"""
from dataclasses import dataclass as _dataclass
import numpy as _np
from math import cos, sin


class Coordinates:
    """Particle coordinates in 6D phase space.

    Follows Zgoubi's convention.

    Examples:
        >>> c = Coordinates()
        >>> c.y
        0.0
        >>> c = Coordinates(0.0,1.0,0.0,1.0,0.0,0.0)
        >>> c.t
        1.0
    """

    def __init__(self, x: float = 0, y: float = 0, z: float = 0, t: float = 0, p: float = 0, d: float = 1):
        self.d = d
        self.x = x
        self.y = y
        self.t = t
        self.z = z
        self.p = p
        self.iex = 1

    def __len__(self) -> int:
        return len(self.list)

    def __eq__(self, other) -> bool:
        return self.list == other.list

    def __repr__(self):
        return str(self.list)

    @property
    def array(self) -> _np.array:
        """Provides a numpy array."""
        return _np.array(self.list)

    @property
    def list(self) -> list:
        """Provides a flat list."""
        return [self.x,
                self.y,
                self.z,
                self.t,
                self.p,
                self.d]

    @classmethod
    def from_list(cls, coords_list):
        return cls(*coords_list)

    def u(self) -> _np.array:
        cos_p = cos(self.p)
        u_x = cos_p * cos(self.t)
        u_y = cos_p * sin(self.t)
        u_z = sin(self.p)
        return _np.array((u_x, u_y, u_z))

    def cartesian(self) -> _np.array:
        return _np.array((self.x, self.y, self.z), float)
