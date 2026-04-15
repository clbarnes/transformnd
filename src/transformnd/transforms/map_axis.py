import numpy as np
from numpy.typing import ArrayLike

from ..base import Transform
from ..util import is_square, none_eq, SpaceTuple

class MapAxis(Transform):
    """Map coordinates from one axis to another.
    
    For example, x -> y and y -> x"""

    #ndim: Optional[Set[int]] = set(2)

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
        self.spaces = spaces

    def apply(self, coords: np.ndarray) -> np.ndarray:
        """Apply transformation to coordinates.
        
        For example: 
        2-D with permutation [1, 0] will give you
        [[x1, y1], [x2, y2]] -> [[y1, x1], [y2, x2]]
        """
        return coords[:, self.permutation]

    def __invert__(self) -> Transform:
        """Invert transformation if possible.

        Returns
        -------
        Transform
            Inverted transformation.
        """
        return np.argsort(self.permutation)