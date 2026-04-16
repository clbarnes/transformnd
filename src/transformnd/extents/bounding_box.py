from functools import lru_cache
from array_api_compat import array_namespace
from ..util import ArrayT, are_coords
from .base import Extents
from array_api_compat import device as xp_device


class BoundingBox(Extents[ArrayT]):
    def __init__(self, mins: ArrayT, maxes: ArrayT) -> None:
        xp = array_namespace(mins)
        if xp.shape(mins) != xp.shape(maxes):
            raise ValueError("mins and maxes must be the same shape")
        if len(xp.shape(mins)) != 1:
            raise ValueError("mins and maxes must be 1D")

        self.ndim = {xp.shape(mins)[0]}
        self.mins = mins
        self.maxes = maxes

    @lru_cache()
    def extents_cast(self, namespace, device) -> tuple[ArrayT, ArrayT]:
        return (
            namespace.asarray(self.mins, device=device),
            namespace.asarray(self.maxes, device=device),
        )

    def _validate_coords(self, coords: ArrayT) -> ArrayT:
        return are_coords(coords, self.ndim)

    def contains(self, coords: ArrayT) -> ArrayT:
        coords = self._validate_coords(coords)
        xp = array_namespace(coords)
        device = xp_device(coords)
        mins, maxes = self.extents_cast(xp, device)

        return xp.logical_and(xp.greater_equal(coords, mins), xp.less(coords, maxes))
