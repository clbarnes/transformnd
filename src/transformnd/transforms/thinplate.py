"""@public
Thin plate splines transformations.

Requires the `thinplatesplines` extra.
"""

import logging

import morphops as mops
import numpy as np

from ..base import Transform
from ..util import as_floats
from ..types import Spaces, NDims

logger = logging.getLogger(__name__)


class ThinPlateSplines(Transform[np.ndarray]):
    """Thin plate splines transforms.

    Deform based on matched pairs of control points.
    """

    def __init__(
        self,
        source_control_points: np.ndarray,
        target_control_points: np.ndarray,
        *,
        spaces: Spaces = Spaces(None, None),
    ):
        """Non-rigid control point based transforms in 2/3D.

        Adapted from
        https://github.com/schlegelp/navis/blob/master/navis/transforms/thinplate.py

        Parameters
        ----------
        source_control_points
            NxD array of control point coordinates in the source space.
        target_control_points
            NxD array of control point coordinates in the target (deformed) space.
        spaces
            Optional source and target spaces

        Raises
        ------
        ValueError
            Invalid control points.
        """
        self.source_control_points = as_floats(source_control_points)
        self.target_control_points = as_floats(target_control_points)

        if self.source_control_points.shape != self.target_control_points.shape:
            raise ValueError("Control point arrays must be the same shape")

        if self.source_control_points.ndim != 2:
            raise ValueError("Control points array must be 2D")

        ndim = self.source_control_points.shape[1]

        self.W, self.A = mops.tps_coefs(
            self.source_control_points,
            self.target_control_points,
        )
        super().__init__(NDims(ndim, ndim), spaces=spaces)

    def invert(self) -> Transform[np.ndarray] | None:
        return type(self)(
            self.target_control_points,
            self.source_control_points,
            spaces=self.spaces.invert(),
        )

    def apply(self, coords: np.ndarray) -> np.ndarray:
        coords = self._validate_coords(coords)
        U = mops.K_matrix(coords, self.source_control_points)
        P = mops.P_matrix(coords)
        # The warped pts are the affine part + the non-uniform part
        return P @ self.A + U @ self.W
