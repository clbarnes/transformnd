"""Thin plate splines transformations."""
import logging
from typing import Optional

import morphops as mops
import numpy as np

from ..base import Transform
from ..util import SpaceRef, check_ndim

logger = logging.getLogger(__name__)

try:
    from scipy.spatial.distance import cdist

    # Replace morphops's original slow distance_matrix function
    mops.lmk_util.distance_matrix = cdist
except ImportError:
    logger.warning(
        "scipy not present; morphops-based transformations may be slower than necessary"
    )


class ThinPlateSplines(Transform):
    ndim = {2, 3}

    def __init__(
        self,
        source_control_points: np.ndarray,
        target_control_points: np.ndarray,
        *,
        source_space: Optional[SpaceRef] = None,
        target_space: Optional[SpaceRef] = None,
    ):
        """Non-rigid control point based transforms in 2/3D.

        Adapted from
        https://github.com/schlegelp/navis/blob/master/navis/transforms/thinplate.py

        Parameters
        ----------
        source_control_points : np.ndarray
            NxD array of control point coordinates in the source space.
        target_control_points : np.ndarray
            NxD array of control point coordinates in the target (deformed) space.
        source_space : Optional[SpaceRef]
        target_space : Optional[SpaceRef]

        Raises
        ------
        ValueError
            Invalid control points.
        """
        super().__init__(source_space=source_space, target_space=target_space)
        self.source_control_points = np.asarray(source_control_points)
        self.target_control_points = np.asarray(target_control_points)

        if self.source_control_points.shape != self.target_control_points.shape:
            raise ValueError("Control point arrays must be the same shape")

        if self.source_control_points.ndim != 2:
            raise ValueError("Control points array must be 2D")

        ndim = self.source_control_points.shape[1]
        check_ndim(ndim, self.ndim)
        self.ndim = {ndim}

        self.W, self.A = mops.tps_coefs(
            self.source_control_points,
            self.target_control_points,
        )

    def __neg__(self) -> Transform:
        return ThinPlateSplines(
            self.target_control_points,
            self.source_control_points,
            source_space=self.target_space,
            target_space=self.source_space,
        )

    def __call__(self, coords: np.ndarray) -> np.ndarray:
        coords = self._validate_coords(coords)
        U = mops.K_matrix(coords, self.source_control_points)
        P = mops.P_matrix(coords)
        # The warped pts are the affine part + the non-uniform part
        return P @ self.A + U @ self.W
