from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from .base import Transform
from .util import SpaceRef, flatten


class LinearMapTransform(Transform):
    def __init__(
        self,
        matrix: ArrayLike,
        *,
        source_space: Optional[SpaceRef] = None,
        target_space: Optional[SpaceRef] = None,
    ):
        super().__init__(source_space=source_space, target_space=target_space)
        self.matrix = np.asarray(matrix)
        if self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Transformation matrix must be 2D and square")
        self.ndim = {len(self.matrix)}

    def __call__(self, coords: np.ndarray) -> np.ndarray:
        self._check_ndim(coords)
        flat, unflatten = flatten(coords)
        return unflatten(self.matrix @ flat)

    def __neg__(self) -> Transform:
        return type(self)(
            np.linalg.inv(self.matrix),
            source_space=self.target_space,
            target_space=self.source_space,
        )


class AffineTransform(LinearMapTransform):
    def __init__(
        self,
        matrix: ArrayLike,
        *,
        source_space: Optional[SpaceRef],
        target_space: Optional[SpaceRef],
    ):
        super().__init__(matrix, source_space=source_space, target_space=target_space)
        self.ndim = {len(self.matrix) - 1}

    def __call__(self, coords: np.ndarray) -> np.ndarray:
        self._check_ndim(coords)
        coords = np.concatenate(
            [coords, np.ones((1,) + coords.shape[1:], dtype=coords.dtype)], axis=0
        )
        flat, unflatten = flatten(coords)
        return unflatten(self.matrix @ flat)

    @classmethod
    def from_linear_map(
        cls,
        linear_map: ArrayLike,
        translation=None,
        *,
        source_space: Optional[SpaceRef] = None,
        target_space: Optional[SpaceRef] = None,
    ):
        if isinstance(linear_map, LinearMapTransform):
            lin_map = linear_map.matrix
        else:
            lin_map = np.asarray(linear_map)

        side = len(lin_map) + 1
        matrix = np.zeros(shape=(side, side), dtype=lin_map.dtype)
        matrix[-1, -1] = 1

        if translation is not None:
            matrix[:-1, -1] = translation

        return cls(matrix, source_space=source_space, target_space=target_space)
