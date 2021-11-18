"""
Simple transformations like rigid translation and scaling.
"""
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from ..base import Transform
from ..util import SpaceRef, chain_or


class IdentityTransform(Transform):
    def __init__(
        self,
        *,
        source_space: Optional[SpaceRef] = None,
        target_space: Optional[SpaceRef] = None,
    ):
        """
        Transform which does nothing.

        Parameters
        ----------
        source_space : Optional[SpaceRef]
        target_space : Optional[SpaceRef]

        Raises
        ------
        ValueError
            [description]
        """
        src = chain_or(source_space, target_space, default=None)
        tgt = chain_or(target_space, source_space, default=None)
        if src != tgt:
            raise ValueError("Source and target spaces are different")
        super().__init__(source_space=src, target_space=src)

    def __neg__(self) -> Transform:
        return self

    def __call__(self, coords: np.ndarray) -> np.ndarray:
        return coords.copy()


class Translate(Transform):
    def __init__(
        self,
        translation: ArrayLike,
        *,
        source_space: Optional[SpaceRef] = None,
        target_space: Optional[SpaceRef] = None,
    ):
        """Simple translation.

        Parameters
        ----------
        translation : scalar or D-length array
            Translation to apply in all dimensions, or each dimension.
        source_space : Optional[SpaceRef]
        target_space : Optional[SpaceRef]

        Raises
        ------
        ValueError
            If the translation is the wrong shape
        """
        super().__init__(source_space=source_space, target_space=target_space)
        self.translation = np.asarray(translation)
        if self.translation.ndim > 1:
            raise ValueError("Translation must be scalar or 1D")

        if self.translation.shape not in [(), (1,)]:
            self.ndim = {self.translation.shape[0]}
        # otherwise, can be broadcast to anything

    def __call__(self, coords: np.ndarray) -> np.ndarray:
        self._validate_coords(coords)
        return coords + self.translation

    def __neg__(self) -> Transform:
        return type(self)(
            -self.translation,
            source_space=self.target_space,
            target_space=self.source_space,
        )


class Scale(Transform):
    def __init__(
        self,
        scale: ArrayLike,
        *,
        source_space: Optional[SpaceRef] = None,
        target_space: Optional[SpaceRef] = None,
    ):
        """Simple scale transform.

        All points are scaled, i.e. distance from the origin may also change.

        Parameters
        ----------
        scale : scalar or D-length array-like
            Scaling to apply in all dimensions, or each dimension.
        source_space : Optional[SpaceRef]
        target_space : Optional[SpaceRef]

        Raises
        ------
        ValueError
            If scale is the wrong shape.
        """
        super().__init__(source_space=source_space, target_space=target_space)
        self.scale = np.asarray(scale)
        if self.scale.ndim > 1:
            raise ValueError("Scale must be scalar or 1D")

        if self.scale.shape not in [(), (1,)]:
            self.ndim = {self.scale.shape[0]}
        # otherwise, can be broadcast to anything

    def __call__(self, coords: np.ndarray) -> np.ndarray:
        coords = self._validate_coords(coords)
        return coords * self.scale

    def __neg__(self) -> Transform:
        return type(self)(
            1 / self.scale,
            source_space=self.target_space,
            target_space=self.source_space,
        )
