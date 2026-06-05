from types import ModuleType
from typing import Self

from scipy.ndimage import map_coordinates
import numpy as np
from array_api_compat import get_namespace

from ..types import NDims, Spaces
from ..base import Transform, ArrayT
from ..util import set_scipy_array_api

set_scipy_array_api()


class Coordinates(Transform[ArrayT]):
    def __init__(
        self,
        coordinates: ArrayT,
        interpolation_order: int = 3,
        *,
        spaces: Spaces = Spaces(None, None),
    ):
        """Use the input coordinates as array indices to look up output coordinates.

        For input coordinate `(a, b, c)`, the output coordinate is `coordinates[a, b, c, :]`.

        Input coordinates outside of the `coordinates` array return `NaN` output coordinates.

        Parameters
        ----------
        coordinates
            Array with `Di + 1` dimensions, where `Di` is the input dimensionality.
            The last dimension's length is the output dimensionality.
        interpolation_order
            Order of the spline interpolation used for coordinates which are not integer array indices.
        spaces
            References for source and target spaces
        """
        xp = get_namespace(coordinates)
        sh = xp.shape(coordinates)
        in_ndim = len(sh) - 1
        out_ndim = sh[-1]
        self.coordinates = coordinates
        self._mode = "constant"
        self._cval = np.nan
        self._order = interpolation_order
        super().__init__(NDims(in_ndim, out_ndim), spaces=spaces)

    def apply(self, coords: ArrayT) -> ArrayT:
        coords = self._validate_coords(coords)
        out = map_coordinates(
            self.coordinates,
            coords,
            order=self._order,
            mode=self._mode,
            cval=self._cval,
        )
        raise NotImplementedError

    def to_device(self, xp: ModuleType, device: str | None = None) -> Self:
        coords = xp.asarray(self.coordinates, device)
        return type(self)(coords, spaces=self.spaces)
