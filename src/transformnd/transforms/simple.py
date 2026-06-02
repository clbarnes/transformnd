"""
Simple transformations like rigid translation and scaling.
"""

from copy import copy
from typing import Self

from numpy.typing import ArrayLike

from array_api_compat import array_namespace
from array_api_compat import device as xp_device
from ..base import Transform
from ..types import NDims, Spaces
from ..util import ArrayT, chain_or, as_floats
from ..transforms.affine import Affine


class Identity(Transform[ArrayT]):
    """No-op transformation."""

    def __init__(
        self,
        ndim: int,
        *,
        spaces: Spaces = Spaces(None, None),
    ):
        """
        Transform which does nothing.

        Parameters
        ----------
        ndim:
            Number of dimensions of this transform.
        spaces:
            Optional source and target spaces
        """
        src = chain_or(*spaces, default=None)
        tgt = chain_or(*spaces[::-1], default=None)
        super().__init__(NDims(ndim, ndim), spaces=Spaces(src, tgt))

    def invert(self) -> Transform[ArrayT]:
        return type(self)(self.ndims.source, spaces=self.spaces.invert())

    def to_affine(self) -> Affine[ArrayT] | None:
        return Affine[ArrayT].identity(self.ndims.source, spaces=self.spaces)

    def apply(self, coords: ArrayT) -> ArrayT:
        return coords


class Translate(Transform[ArrayT]):
    """Translate coordinates by addition."""

    def __init__(
        self,
        translation: ArrayLike,
        *,
        spaces: Spaces = Spaces(None, None),
    ):
        """Simple translation.

        Parameters
        ----------
        translation : D-length array
            Translation to apply in all dimensions, or each dimension.
        spaces : tuple[SpaceRef, SpaceRef]
            Optional source and target spaces

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
        super().__init__(
            NDims(len(self.translation), len(self.translation)), spaces=spaces
        )

    def to_affine(self) -> Affine[ArrayT] | None:
        return Affine[ArrayT].translation(self.translation, spaces=self.spaces)

    def apply(self, coords: ArrayT) -> ArrayT:
        coords = self._validate_coords(coords)
        xp = array_namespace(coords)
        d = xp_device(coords)
        return coords + xp.asarray(self.translation, device=d)

    def invert(self) -> Transform | None:
        return type(self)(-self.translation, spaces=self.spaces.invert())

    def to_device(self, xp, device=None) -> Self:
        result = copy(self)
        result.translation = xp.asarray(self.translation, device=device)
        return result


class Scale(Transform[ArrayT]):
    """Scale coordinates by multiplication."""

    def __init__(
        self,
        scale: ArrayLike,
        *,
        spaces: Spaces = Spaces(None, None),
    ):
        """Simple scale transform.

        All points are scaled, i.e. distance from the origin may also change.

        Parameters
        ----------
        scale : scalar or D-length array-like
            Scaling to apply in all dimensions, or each dimension.
        spaces : tuple[SpaceRef, SpaceRef]
            Optional source and target spaces

        Raises
        ------
        ValueError
            If scale is the wrong shape.
        """
        self.scale = as_floats(scale)
        if self.scale.ndim != 1:
            raise ValueError(f"Scale must be 1D, got shape {self.scale.shape}")
        super().__init__(NDims(len(self.scale), len(self.scale)), spaces=spaces)

    def to_affine(self) -> Affine[ArrayT] | None:
        return Affine[ArrayT].scaling(self.scale, spaces=self.spaces)

    def apply(self, coords: ArrayT) -> ArrayT:
        coords = self._validate_coords(coords)
        xp = array_namespace(coords)
        d = xp_device(coords)
        return coords * xp.asarray(self.scale, device=d)

    def invert(self) -> Self | None:
        return type(self)(
            1 / self.scale,
            spaces=self.spaces.invert(),
        )

    def to_device(self, xp, device=None) -> Self:
        result = copy(self)
        result.scale = xp.asarray(self.scale, device=device)
        return result
