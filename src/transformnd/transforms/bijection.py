import numpy as np

from ..base import Transform
from ..util import SpaceTuple, dim_intersection


class Bijection(Transform):
    """A bijection is a transormation with a user-assigned inverse.
    This is useful for transformations that are not easily invertible.
    """

    # ndim: Optional[Set[int]] = set(2)

    def __init__(
        self,
        forward: Transform,
        inverse: Transform,
        *,
        spaces: SpaceTuple = (None, None),
    ):
        """Base class for transformations.

        Parameters
        ----------
        spaces : tuple[SpaceRef, SpaceRef]
            Optional source and target spaces
        """

        self.forward = forward
        self.inverse = inverse
        self.spaces = spaces

        ndim = dim_intersection(self.forward.ndim, self.inverse.ndim)

        if ndim is not None and len(ndim) == 0:
            raise ValueError(
                "forward and inverse transforms support different dimensionalities"
            )

        self.ndim = ndim

    def apply(self, coords: np.ndarray) -> np.ndarray:
        return self.forward.apply(coords)

    def __invert__(self) -> Transform:

        return type(self)(self.inverse, self.forward)
