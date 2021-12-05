import field_point
from abc import ABC, abstractmethod


class Field(ABC):
    """Abstract class to represent a field. Child classes are Analytical_field, Mesh_field...
    It can be described by analytical values
    TODO : add grid mesh or extrapolation from median plane

    """

    @abstractmethod
    def __init__(self, descriptor):
        """

        :param descriptor: The descriptor might be a function of x, y, z; A meshgrid or a mesh on a plane.
        """
        self.descriptor = descriptor

    @abstractmethod
    def get_value(self, x: float, y:float, z: float) -> field_point:
        pass
