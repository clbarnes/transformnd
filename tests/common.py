from transformnd.base import Transform
from transformnd.types import Spaces, NDims
from transformnd.transforms.affine import Affine
from copy import copy
import numpy as np


class NullTransform(Transform):
    """Flexible identity-like transform used for testing."""

    def __init__(
        self,
        ndim: int,
        invertible: bool = False,
        affineable: bool = False,
        *,
        spaces: Spaces = Spaces(None, None),
    ):
        super().__init__(NDims(ndim, ndim), spaces=spaces)
        self.invertible = invertible
        self.affineable = affineable

    def invert(self) -> Transform | None:
        if self.invertible:
            return copy(self)
        return None

    def to_affine(self) -> Affine | None:
        if self.affineable:
            return Affine.identity(self.ndims.source, spaces=self.spaces)
        return None

    def apply(self, coords: np.ndarray) -> np.ndarray:
        return coords.copy()
