from typing import Self

from array_api_compat import array_namespace

from transformnd.transforms.affine import Affine

from ..base import Transform, ArrayT


class Bijection(Transform[ArrayT]):
    """Map coordinates from one axis to another.

    For example, x -> y and y -> x"""

    def __init__(
        self,
        forward: Transform[ArrayT],
        inverse: Transform[ArrayT],
    ):
        """Base class for transformations.

        Parameters
        ----------
        forward
            The forward transformation.
        inverse
            The inverse transformation.

        Raises
        ------
        ValueError
            If the forward and inverse dimensionalities don't match.
        """
        self.forward = forward
        self.inverse = inverse
        if forward.ndims != inverse.ndims.invert():
            raise ValueError(
                f"Bijection dimensionalities mismatch: fwd:{forward.ndims}, inv:{inverse.ndims}"
            )
        super().__init__(self.forward.ndims)

    def apply(self, coords: ArrayT) -> ArrayT:
        return self.forward.apply(coords)

    def invert(self) -> Self | None:
        return type(self)(self.inverse, self.forward)

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
