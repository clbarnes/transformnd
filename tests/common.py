from transformnd.base import Transform
from transformnd.types import NDims
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
    ):
        super().__init__(NDims(ndim, ndim))
        self.invertible = invertible
        self.affineable = affineable

    def invert(self) -> Transform | None:
        if self.invertible:
            return copy(self)
        return None

    def to_affine(self) -> Affine | None:
        if self.affineable:
            return Affine.identity(self.ndims.source)
        return None

    def apply(self, coords: np.ndarray) -> np.ndarray:
        return coords.copy()
