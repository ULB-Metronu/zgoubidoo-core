"""This module provides a clean interface to the Zgoubi coordinates system (Y-T-Z-P-X-D).

A dataclass is used for the implementation but conversions to common formats are also provided.
"""
import numpy as _np
from zgoubidoo_core.physics.coordinates import Coordinates


class Particle:
    """Particle representation in 6D phase space.
    """
    def __init__(self, coords: Coordinates = Coordinates(), rigidity: float = 100):
        self.rigidity = rigidity
        self.complete_coords = coords

    def __getitem__(self, item: int):
        return getattr(self, list(self.__dict__.keys())[item])

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

    def u(self) -> _np.array:
        return self.complete_coords.u()

    def set_coords(self, r, u, rigidity):
        pass  # TODO

    def cartesian(self) -> _np.array:
        return self.complete_coords.cartesian()
