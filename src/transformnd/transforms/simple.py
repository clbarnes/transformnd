"""
Simple transformations like rigid translation and scaling.
"""

from copy import copy
from typing import Self

import numpy as np
from numpy.typing import ArrayLike

from array_api_compat import array_namespace
from array_api_compat import device as xp_device
from ..base import Transform
from ..util import ArrayT, chain_or, SpaceTuple, invert_spaces


class Identity(Transform):
    """No-op transformation."""

    def __init__(
        self,
        *,
        spaces: SpaceTuple = (None, None),
    ):
        """
        Transform which does nothing.

        Parameters
        ----------
        spaces : tuple[SpaceRef, SpaceRef]
            Optional source and target spaces

        Raises
        ------
        ValueError
            [description]
        """
        src = chain_or(*spaces, default=None)
        tgt = chain_or(*spaces[::-1], default=None)
        if src != tgt:
            raise ValueError("Source and target spaces are different")
        super().__init__(spaces=(src, src))

    def __invert__(self) -> Transform:
        return self

    def apply(self, coords: ArrayT) -> ArrayT:
        xp = array_namespace(coords)
        return xp.asarray(coords, copy=True)


class Translate(Transform):
    """Translate coordinates by addition."""

    def __init__(
        self,
        translation: ArrayLike,
        *,
        spaces: SpaceTuple = (None, None),
    ):
        """Simple translation.

        Parameters
        ----------
        translation : scalar or D-length array
            Translation to apply in all dimensions, or each dimension.
        spaces : tuple[SpaceRef, SpaceRef]
            Optional source and target spaces

        Raises
        ------
        ValueError
            If the translation is the wrong shape
        """
        super().__init__(spaces=spaces)
        self.translation = np.asarray(translation)
        if self.translation.ndim > 1:
            raise ValueError("Translation must be scalar or 1D")

        if self.translation.shape not in [(), (1,)]:
            self.ndim = {self.translation.shape[0]}
        # otherwise, can be broadcast to anything

    def apply(self, coords: ArrayT) -> ArrayT:
        coords = self._validate_coords(coords)
        xp = array_namespace(coords)
        d = xp_device(coords)
        return coords + xp.asarray(self.translation, device=d)

    def __invert__(self) -> Transform:
        return type(self)(-self.translation, spaces=(self.spaces[1], self.spaces[0]))

    def to_device(self, xp, device=None) -> Self:
        result = copy(self)
        result.translation = xp.asarray(self.translation, device=device)
        return result


class Scale(Transform):
    """Scale coordinates by multiplication."""

    def __init__(
        self,
        scale: ArrayLike,
        *,
        spaces: SpaceTuple = (None, None),
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
        super().__init__(spaces=spaces)
        self.scale = np.asarray(scale)
        if self.scale.ndim > 1:
            raise ValueError("Scale must be scalar or 1D")

        if self.scale.shape not in [(), (1,)]:
            self.ndim = {self.scale.shape[0]}
        # otherwise, can be broadcast to anything

    def apply(self, coords: ArrayT) -> ArrayT:
        coords = self._validate_coords(coords)
        xp = array_namespace(coords)
        d = xp_device(coords)
        return coords * xp.asarray(self.scale, device=d)

    def __invert__(self) -> Transform:
        return type(self)(
            1 / self.scale,
            spaces=invert_spaces(self.spaces),
        )

    def to_device(self, xp, device=None) -> Self:
        result = copy(self)
        result.scale = xp.asarray(self.scale, device=device)
        return result
