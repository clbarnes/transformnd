from transformnd.base import Transform
from transformnd.util import SpaceTuple, to_single_ndim
from transformnd.transforms.affine import Affine
from copy import copy
import numpy as np


class NullTransform(Transform):
    """Flexible identity-like transform used for testing."""

    def __init__(
        self,
        ndim: set[int] | None = None,
        invertible: bool = False,
        affineable: bool = False,
        *,
        spaces: SpaceTuple = (None, None),
    ):
        super().__init__(spaces=spaces)
        self.ndim = ndim
        self.invertible = invertible
        self.affineable = affineable

    def invert(self) -> Transform | None:
        if self.invertible:
            return copy(self)
        return None

    def to_affine(self, ndim: int | None = None) -> Affine | None:
        ndim = to_single_ndim(ndim, self.ndim)
        if self.affineable:
            return Affine.identity(ndim, spaces=self.spaces)
        return None

    def apply(self, coords: np.ndarray) -> np.ndarray:
        return coords.copy()
