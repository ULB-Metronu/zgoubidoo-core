"""This module provides a clean interface to the Zgoubi coordinates system (Y-T-Z-P-X-D).

A dataclass is used for the implementation but conversions to common formats are also provided.
"""
from dataclasses import dataclass as _dataclass
import numpy as _np
from coordinates import Coordinates


@_dataclass
class Particle:
    """Particle coordinates in 6D phase space.

    Follows Zgoubi's convention.

    Examples:
        >>> c = Particle()
        >>> c.y
        0.0
        >>> c = Coordinates(1.0, 1.0, 0.0, 0.0, 0.0, 0.0)
        >>> c.t
        1.0
    """
    coords: Coordinates
    """Coordinates of the particle"""
    rigidity: float
    """Rigidity of the particle (T*m)"""

    def __getitem__(self, item: int):
        return getattr(self, list(self.__dataclass_fields__.keys())[item])

    def __eq__(self, other) -> bool:
        return self.list == other.list

    @property
    def array(self) -> _np.array:
        """Provides a numpy array."""
        return _np.array(self.list)

    @property
    def list(self) -> list:
        """Provides a flat list."""
        return list(self.__dict__.values())

    @property
    def rigidity(self) -> float:
        return self.rigidity

    def u(self) -> _np.array:
        return self.coords.u()

    def set_coords(self, r, u, rigidity):
        pass  # TODO

    def cartesian(self) -> _np.array:
        return self.coords.cartesian()
