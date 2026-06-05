import logging

import numpy as np
import shapely
from shapely.geometry.base import BaseGeometry
from shapely.coords import CoordinateSequence

from ..base import Transform, ArrayT
from .base import BaseAdapter

logger = logging.getLogger(__name__)


def as_numpy(coords: CoordinateSequence) -> np.ndarray:
    return np.asarray(coords)


class GeometryAdapter(BaseAdapter[BaseGeometry, ArrayT]):
    """Transform shapely geometries.

    As well as the generic `apply()`,
    there are `apply_*()` methods for transforming different geometry subclasses.

    N.B. some transforms may create invalid topologies
    (incorrect winding, self-intersections etc.).

    N.B. shapely geometries' coordinates are in `XY(Z)(M)` order
    """

    def apply[T: BaseGeometry](
        self,
        transform: Transform,
        obj: T,
        *,
        include_z: bool | None = None,
    ) -> T:
        """Transform the shapely geometry.

        Parameters
        ----------
        transform
            The transformation to apply.
        obj
            Some shapely geometry in 2 or 3D
        include_z
            Force inclusion/ exclusion of Z coordinate.
            By default (None), checks whether the given geometry has Z coordinates.

        Returns
        -------
        T
            An object of the same type as the input.
        """

        def fn(coords: np.ndarray) -> np.ndarray:
            c = coords.copy()
            return transform.apply(c)

        if include_z is None:
            inc_z = bool(shapely.has_z(obj))
        else:
            inc_z = include_z

        return shapely.transform(obj, fn, include_z=inc_z)
