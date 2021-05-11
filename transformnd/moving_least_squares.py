from typing import Optional

import numpy as np
from molesq.transform import Transformer as _Transformer

from .base import Transform
from .util import SpaceRef


class MovingLeastSquares(Transform):
    def __init__(
        self,
        source_control_points: np.ndarray,
        target_control_points: np.ndarray,
        *,
        source_space: Optional[SpaceRef] = None,
        target_space: Optional[SpaceRef] = None
    ):
        super().__init__(source_space=source_space, target_space=target_space)
        self._transformer = _Transformer(
            np.asarray(source_control_points),
            np.asarray(target_control_points),
        )
        self.ndim = {self._transformer.control_points.shape[1]}

    def __call__(self, coords: np.ndarray) -> np.ndarray:
        coords = self._validate_coords(coords)
        return self._transformer.transform(coords)

    def __neg__(self) -> Transform:
        return MovingLeastSquares(
            self._transformer.deformed_control_points,
            self._transformer.control_points,
            source_space=self.target_space,
            target_space=self.source_space,
        )
