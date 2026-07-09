from typing import Self
from array_api_compat import array_namespace
import numpy as np

from ..base import Transform
from ..util import ArrayT, join_strs
from ..types import NDims
from ..transforms.affine import Affine


class MapAxis(Transform[ArrayT]):
    """Map coordinates from one axis to another.

    For example, x -> y and y -> x"""

    def __init__(
        self,
        permutation: list[int],
    ):
        """Base class for transformations.

        Parameters
        ----------
        permutation
            New order of column axis. For example, [1, 0] means x -> y and y -> x.

        Raises
        ------
        ValueError
            If permutation does not contain all dimensions [0, N) exactly once.
        """
        s_perm = sorted(permutation)
        if any(a != b for a, b in enumerate(s_perm)):
            raise ValueError(
                "N-D permutation must contain all dimensions [0, N) exactly once"
            )
        self.permutation = permutation
        super().__init__(NDims(len(permutation), len(permutation)))

    def is_identity(self) -> bool:
        return all(a == b for a, b in enumerate(self.permutation))

    def to_affine(self) -> Affine[ArrayT] | None:
        m = np.eye(self.ndims.source)
        m = m[self.permutation, :]
        return Affine.from_linear_map(m)  # type: ignore

    def apply(self, coords: ArrayT) -> ArrayT:
        """Apply transformation to coordinates.

        For example:
        2-D with permutation [1, 0] will give you
        [[x1, y1], [x2, y2]] -> [[y1, x1], [y2, x2]]
        """

        coords = self._validate_coords(coords)
        xp = array_namespace(coords)
        return xp.take(coords, self.permutation, 1)

    def invert(self) -> Self | None:
        return type(self)(
            list(np.argsort(self.permutation)),
        )

    def __str__(self) -> str:
        return f"{super().__str__()}({join_strs(self.permutation)})"
