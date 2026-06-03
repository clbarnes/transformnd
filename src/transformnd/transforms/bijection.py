from typing import Self

from array_api_compat import array_namespace

from transformnd.transforms.affine import Affine

from ..base import Transform, ArrayT
from ..types import Spaces
from ..util import same_or_none


class Bijection(Transform[ArrayT]):
    """Map coordinates from one axis to another.

    For example, x -> y and y -> x"""

    def __init__(
        self,
        forward: Transform[ArrayT],
        inverse: Transform[ArrayT],
        *,
        spaces: Spaces = Spaces(None, None),
    ):
        """Base class for transformations.

        Parameters
        ----------
        spaces : tuple[SpaceRef, SpaceRef]
            Optional source and target spaces
        """
        src = same_or_none(
            spaces.source, forward.spaces.source, inverse.spaces.target, default=None
        )
        tgt = same_or_none(
            spaces.target, forward.spaces.target, inverse.spaces.source, default=None
        )

        self.forward = forward
        self.inverse = inverse
        if forward.ndims != inverse.ndims.invert():
            raise ValueError(
                f"Bijection dimensionalities mismatch: fwd:{forward.ndims}, inv:{inverse.ndims}"
            )
        super().__init__(self.forward.ndims, spaces=Spaces(src, tgt))

    def apply(self, coords: ArrayT) -> ArrayT:
        return self.forward.apply(coords)

    def invert(self) -> Self | None:
        return type(self)(self.inverse, self.forward, spaces=self.spaces.invert())

    def is_identity(self) -> bool:
        return self.forward.is_identity() and self.inverse.is_identity()

    def to_affine(self) -> Affine[ArrayT] | None:
        fwd = self.forward.to_affine()
        if fwd is None:
            return None
        inv = self.inverse.to_affine()
        if inv is None:
            return None

        inv_inv = inv.invert()
        if inv_inv is None:
            return None

        xp = array_namespace(fwd.matrix)
        if xp.equal(fwd.matrix, inv_inv.matrix):
            return fwd

        return None
