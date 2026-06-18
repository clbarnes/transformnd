from typing import Any, Protocol, Generic

from array_api_compat import array_namespace
from transformnd.types import NDims, Spaces
from ..base import Transform
from ..types import ArrayT


class Interpolator(Protocol, Generic[ArrayT]):
    def __call__(self, x: ArrayT) -> ArrayT: ...


class GridInterpolation(Transform):
    def __init__(
        self, interpolators: list[Interpolator], *, spaces: Spaces = Spaces(None, None)
    ):
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
