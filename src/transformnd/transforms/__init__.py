"""Implementations of some common transforms."""

from .affine import Affine
from .reflection import Reflect
from .simple import Identity, Scale, Translate
from .map_axis import MapAxis
from .bijection import Bijection
from .project_axis import ProjectAxis, Insert, Remove
from .by_dimension import ByDimension, SubTransform
from .vector_field import Coordinates, Displacements
from .moving_least_squares import MovingLeastSquares
from .thinplate import ThinPlateSplines

__all__ = [
    "Affine",
    "Identity",
    "ProjectAxis",
    "Insert",
    "Remove",
    "Reflect",
    "Scale",
    "Translate",
    "MapAxis",
    "Bijection",
    "ByDimension",
    "SubTransform",
    "Coordinates",
    "Displacements",
    "MovingLeastSquares",
    "ThinPlateSplines",
]
