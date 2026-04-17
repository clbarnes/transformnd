from array_api_compat import array_namespace
import numpy as np

from ..base import Transform
from ..util import ArrayT, SpaceTuple


class MapAxis(Transform[ArrayT]):
    """Map coordinates from one axis to another.

    For example, x -> y and y -> x"""

    # ndim: Optional[Set[int]] = set(2)

    def __init__(
        self,
        permutation: list[int],
        *,
        spaces: SpaceTuple = (None, None),
    ):
        """Base class for transformations.

        Parameters
        ----------
        permutation: list[int]
            New order of column axis. For example, [1, 0] means x -> y and y -> x.
        spaces : tuple[SpaceRef, SpaceRef]
            Optional source and target spaces
        """
        self.permutation = permutation
        self.ndim = {len(permutation)}
        self.spaces = spaces

    def apply(self, coords: ArrayT) -> ArrayT:
        """Apply transformation to coordinates.

        For example:
        2-D with permutation [1, 0] will give you
        [[x1, y1], [x2, y2]] -> [[y1, x1], [y2, x2]]
        """

        coords = self._validate_coords(coords)
        xp = array_namespace(coords)
        return xp.take(coords, self.permutation, 1)

    def __invert__(self) -> Transform[ArrayT]:
        """Invert transformation if possible.

        Returns
        -------
        Transform[ArrayT]
            Inverted transformation.
        """
        return type(self)(
            list(np.argsort(self.permutation)),
            spaces=(self.spaces[1], self.spaces[0]),
        )
