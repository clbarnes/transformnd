"""
Implementation of Moving Least Squares transformation.

Powered by molesq, an optional dependency.
"""
import numpy as np
from molesq.transform import Transformer as _Transformer

from ..base import SpaceTuple, Transform


class MovingLeastSquares(Transform):
    def __init__(
        self,
        source_control_points: np.ndarray,
        target_control_points: np.ndarray,
        *,
        spaces: SpaceTuple = (None, None),
    ):
        """Non-rigid transforms powered by molesq package.

        Parameters
        ----------
        source_control_points : np.ndarray
            NxD array of control point coordinates in the source space.
        target_control_points : np.ndarray
            NxD array of coordinates of the corresponding control points
            in the target (deformed) space.
        spaces : tuple[SpaceRef, SpaceRef]
            Optional source and target spaces
        """
        super().__init__(spaces=spaces)
        self._transformer = _Transformer(
            np.asarray(source_control_points),
            np.asarray(target_control_points),
        )
        self.ndim = {self._transformer.control_points.shape[1]}

    def __call__(self, coords: np.ndarray) -> np.ndarray:
        coords = self._validate_coords(coords)
        return self._transformer.transform(coords)

    def __invert__(self) -> Transform:
        return MovingLeastSquares(
            self._transformer.deformed_control_points,
            self._transformer.control_points,
            spaces=self.spaces[::-1],
        )
