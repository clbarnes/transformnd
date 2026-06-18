from typing import Any, Protocol, Generic

from array_api_compat import array_namespace
from transformnd.types import NDims, Spaces
from ..base import Transform
from ..types import ArrayT


class Interpolator(Protocol, Generic[ArrayT]):
    def __call__(self, x: ArrayT) -> ArrayT: ...


class GridInterpolation(Transform[ArrayT]):
    """Coordinate transformation which applies a callable to each dimension.

    Intended for use with instances of `scipy.interpolate` interpolators,
    but any callable which takes and returns an array of floats would work.
    """

    def __init__(
        self,
        interpolators: list[Interpolator[ArrayT]],
        *,
        spaces: Spaces = Spaces(None, None),
    ):
        """
        Parameters
        ----------
        interpolators
            One callable per dimension, in order.
            Each one should take and return an array of floats.
        spaces
            Source and target space identifiers
        """
        self.interpolators = interpolators
        nd = len(interpolators)
        super().__init__(NDims(nd, nd), spaces=spaces)

    def apply(self, coords: Any) -> Any:
        xp = array_namespace(coords)
        coords = self._validate_coords(coords)
        coords_t = xp.transpose(coords)
        out_coords_t = xp.zeros_like(coords_t)
        for in_col, out_col, interp in zip(coords_t, out_coords_t, self.interpolators):
            out_col[:] = interp(in_col)

        return xp.transpose(out_coords_t)
