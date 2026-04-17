from ..base import Transform, ArrayT
from ..util import SpaceTuple, dim_intersection, invert_spaces


class Bijection(Transform[ArrayT]):
    """Map coordinates from one axis to another.

    For example, x -> y and y -> x"""

    # ndim: Optional[Set[int]] = set(2)

    def __init__(
        self,
        forward: Transform[ArrayT],
        inverse: Transform[ArrayT],
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
        ndim = dim_intersection(forward.ndim, inverse.ndim)
        if ndim is not None and len(ndim) == 0:
            raise ValueError(
                "forward and inverse transforms do not share a dimensionality"
            )
        self.ndim = ndim
        self.spaces = spaces

    def apply(self, coords: ArrayT) -> ArrayT:
        return self.forward.apply(coords)

    def __invert__(self) -> Transform[ArrayT]:
        return type(self)(self.inverse, self.forward, spaces=invert_spaces(self.spaces))
