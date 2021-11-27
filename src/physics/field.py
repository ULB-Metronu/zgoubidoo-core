"""This module provides a clean interface to the Zgoubi coordinates system (Y-T-Z-P-X-D).

A dataclass is used for the implementation but conversions to common formats are also provided.
"""
from dataclasses import dataclass as _dataclass
import numpy as _np


@_dataclass
class Field:
    """Particle coordinates in 6D phase space.

    Follows Zgoubi's convention.

    Examples:
        >>> c = Field()
        >>> c.y
        0.0
        >>> c = Field(1.0, 1.0, 0.0)
        >>> c.t
        1.0
    """
    x: float = 0
    """Longitudinal coordinate."""
    y: float = 0
    """Horizontal plane coordinate."""
    z: float = 0
    """Vertical plane coordinate"""

    def __getitem__(self, item: int):
        return getattr(self, list(self.__dataclass_fields__.keys())[item])

    def __len__(self) -> int:
        return len(self.list)

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
