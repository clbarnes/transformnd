"""
Simple transformations like rigid translation and scaling.
"""
import numpy as np
from numpy.typing import ArrayLike

from ..base import SpaceTuple, Transform
from ..util import chain_or


class IdentityTransform(Transform):
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
        tgt = chain_or(*spaces, default=None)
        if src != tgt:
            raise ValueError("Source and target spaces are different")
        super().__init__(spaces=(src, src))

    def __invert__(self) -> Transform:
        return self

    def __call__(self, coords: np.ndarray) -> np.ndarray:
        return coords.copy()


class Translate(Transform):
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

    def __call__(self, coords: np.ndarray) -> np.ndarray:
        self._validate_coords(coords)
        return coords + self.translation

    def __invert__(self) -> Transform:
        return type(self)(-self.translation, spaces=self.spaces[::-1])


class Scale(Transform):
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

    def __call__(self, coords: np.ndarray) -> np.ndarray:
        coords = self._validate_coords(coords)
        return coords * self.scale

    def __invert__(self) -> Transform:
        return type(self)(
            1 / self.scale,
            spaces=self.spaces[::-1],
        )
