""" defines Geometric objects with Medium properties """

from .base import Tidy3dBaseModel
from .geometry import GeometryType
from .medium import MediumType
from .types import Axis, AxesSubplot


class Structure(Tidy3dBaseModel):
    """An object that interacts with the electromagnetic fields"""

    geometry: GeometryType
    medium: MediumType

    def plot(  # pylint: disable=invalid-name
        self, position: float, axis: Axis, ax=None, **plot_params: dict
    ) -> AxesSubplot:
        """plot geometry"""
        return self.geometry.plot(position=position, axis=axis, ax=ax, **plot_params)
