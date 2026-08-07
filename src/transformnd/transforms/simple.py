"""
Simple transformations like rigid translation and scaling.
"""

from copy import copy
from typing import Self
from types import ModuleType

from numpy.typing import ArrayLike

from array_api_compat import array_namespace
from array_api_compat import device as xp_device
from ..base import Transform
from ..types import NDims
from ..util import ArrayT, as_floats, join_strs
from ..transforms.affine import Affine


class Identity(Transform[ArrayT]):
    """No-op transformation."""

    def __init__(
        self,
        ndim: int,
    ):
        """
        Transform which does nothing.

        Parameters
        ----------
        ndim:
            Number of dimensions of this transform.
        """
        super().__init__(NDims(ndim, ndim))

    def invert(self) -> Transform[ArrayT]:
        return type(self)(self.ndims.source)

    def to_affine(self) -> Affine[ArrayT]:
        return Affine[ArrayT].identity(self.ndims.source)

    def apply(self, coords: ArrayT) -> ArrayT:
        return coords

    def __str__(self) -> str:
        return f"{super().__str__()}({self.ndims.source})"


class Translate(Transform[ArrayT]):
    """Translate coordinates by addition."""

    def __init__(
        self,
        translation: ArrayLike,
    ):
        """Simple translation.

        Parameters
        ----------
        translation
            Translation to apply in all dimensions, or each dimension.

        Raises
        ------
        ValueError
            If the translation is the wrong shape
        """
        self.translation = as_floats(translation)
        if self.translation.ndim != 1:
            raise ValueError(
                f"Translation must be 1D, got shape {self.translation.shape}"
            )
        super().__init__(NDims(len(self.translation), len(self.translation)))

    def to_affine(self) -> Affine[ArrayT]:
        return Affine[ArrayT].translation(self.translation)

    def apply(self, coords: ArrayT) -> ArrayT:
        coords = self._validate_coords(coords)
        xp = array_namespace(coords)
        d = xp_device(coords)
        return coords + xp.asarray(self.translation, device=d)

    def invert(self) -> Transform | None:
        return type(self)(-self.translation)

    def to_device(self, xp: ModuleType, device: str | None = None) -> Self:
        result = copy(self)
        result.translation = xp.asarray(self.translation, device=device)
        return result

    def __str__(self) -> str:
        return f"{super().__str__()}({join_strs(self.translation)})"


class Scale(Transform[ArrayT]):
    """Scale coordinates by multiplication."""

    def __init__(
        self,
        scale: ArrayLike,
    ):
        """Simple scale transform.

        All points are scaled, i.e. distance from the origin may also change.

        Parameters
        ----------
        scale
            Scaling to apply in all dimensions, or each dimension.

        Raises
        ------
        ValueError
            If scale is the wrong shape.
        """
        self.scale = as_floats(scale)
        if self.scale.ndim != 1:
            raise ValueError(f"Scale must be 1D, got shape {self.scale.shape}")
        super().__init__(NDims(len(self.scale), len(self.scale)))

    def to_affine(self) -> Affine[ArrayT]:
        return Affine[ArrayT].scaling(self.scale)

    def apply(self, coords: ArrayT) -> ArrayT:
        coords = self._validate_coords(coords)
        xp = array_namespace(coords)
        d = xp_device(coords)
        return coords * xp.asarray(self.scale, device=d)

    def invert(self) -> Self | None:
        return type(self)(
            1 / self.scale,
        )

    def to_device(self, xp: ModuleType, device: str | None = None) -> Self:
        result = copy(self)
        result.scale = xp.asarray(self.scale, device=device)
        return result

    def __str__(self) -> str:
        return f"{super().__str__()}({join_strs(self.scale)})"
