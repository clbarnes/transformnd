"""@public
Implementation of Moving Least Squares transformation.

Requires the `movingleastsquares` extra.
"""

from array_api_compat import array_namespace
import numpy as np
from typing import Self
from molesq.transform import Transformer as _Transformer

from ..base import Transform
from ..types import NDims, Spaces
from ..util import as_floats


class MovingLeastSquares(Transform[np.ndarray]):
    """Moving least squares transformation.

    Deform based on a matched pairs of source and target control points; see <https://dl.acm.org/doi/10.1145/1141911.1141920>
    """

    def __init__(
        self,
        source_control_points: np.ndarray,
        target_control_points: np.ndarray,
        *,
        spaces: Spaces = Spaces(None, None),
    ):
        """Non-rigid transforms powered by molesq package.

        Parameters
        ----------
        source_control_points
            NxD array of control point coordinates in the source space.
        target_control_points
            NxD array of coordinates of the corresponding control points
            in the target (deformed) space.
        spaces
            Optional source and target spaces
        """
        s = as_floats(source_control_points)
        t = as_floats(target_control_points)
        self._transformer = _Transformer(s, t)
        super().__init__(
            NDims(
                s.shape[1],
                t.shape[1],
            ),
            spaces=spaces,
        )

    def apply(self, coords: np.ndarray) -> np.ndarray:
        coords = self._validate_coords(coords)
        return self._transformer.transform(coords)

    def is_identity(self) -> bool:
        xp = array_namespace(self._transformer.control_points)
        return xp.all(
            xp.equal(
                self._transformer.control_points,
                self._transformer.deformed_control_points,
            )
        )

    def invert(self) -> Self | None:
        return type(self)(
            self._transformer.deformed_control_points,
            self._transformer.control_points,
            spaces=self.spaces.invert(),
        )
