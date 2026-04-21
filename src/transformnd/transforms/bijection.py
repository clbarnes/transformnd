from typing import Self

from array_api_compat import array_namespace

from transformnd.transforms.affine import Affine

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

    def invert(self) -> Self | None:
        return type(self)(self.inverse, self.forward, spaces=invert_spaces(self.spaces))

    def is_identity(self) -> bool:
        return self.forward.is_identity() and self.inverse.is_identity()

    def to_affine(self, ndim: int | None = None) -> Affine[ArrayT] | None:
        fwd = self.forward.to_affine(ndim)
        if fwd is None:
            return None
        inv = self.inverse.to_affine(ndim)
        if inv is None:
            return None

        inv_inv = inv.invert()
        if inv_inv is None:
            return None

        xp = array_namespace(fwd.matrix)
        if xp.equal(fwd.matrix, inv_inv.matrix):
            return fwd

        return None
