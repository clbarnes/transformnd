from functools import lru_cache
from array_api_compat import array_namespace
from ..util import ArrayT
from .base import Extents
from array_api_compat import device as xp_device


class BoundingBox(Extents[ArrayT]):
    def __init__(self, mins: ArrayT, maxes: ArrayT) -> None:
        xp = array_namespace(mins)
        if xp.shape(mins) != xp.shape(maxes):
            raise ValueError("mins and maxes must be the same shape")
        if len(xp.shape(mins)) != 1:
            raise ValueError("mins and maxes must be 1D")

        self.mins = mins
        self.maxes = maxes
        super().__init__(xp.shape(mins)[0])

    @lru_cache()
    def extents_cast(self, namespace, device) -> tuple[ArrayT, ArrayT]:
        return (
            namespace.asarray(self.mins, device=device),
            namespace.asarray(self.maxes, device=device),
        )

    def _validate_coords(self, coords: ArrayT) -> ArrayT:
        xp = array_namespace(coords)
        if xp.ndim(coords) != 2:
            raise ValueError("Coords must be a 2D array")
        dim = xp.shape(coords)[1]
        if xp.shape(coords)[1] != self.ndim:
            raise ValueError(f"Coords must have dimensionality {self.ndim}, got {dim}")
        return coords

    def contains(self, coords: ArrayT) -> ArrayT:
        coords = self._validate_coords(coords)
        xp = array_namespace(coords)
        device = xp_device(coords)
        mins, maxes = self.extents_cast(xp, device)

        return xp.logical_and(xp.greater_equal(coords, mins), xp.less(coords, maxes))
