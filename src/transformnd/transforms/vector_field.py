from abc import ABC
from collections.abc import Iterable
from types import ModuleType
from typing import Self

from scipy.ndimage import map_coordinates
import numpy as np
from array_api_compat import get_namespace

from ..types import NDims, Spaces
from ..base import Transform, ArrayT
from ..util import set_scipy_array_api

set_scipy_array_api()

__all__ = ["Coordinates", "Displacements"]


class BaseVectorField(Transform[ArrayT], ABC):
    def __init__(
        self,
        vector_field: ArrayT,
        index_transform: Transform[ArrayT] | None = None,
        interpolation_order: int = 3,
        vector_axis: int = -1,
        *,
        spaces: Spaces = Spaces(None, None),
    ):
        """Look up a vector in array.

        Parameters
        ----------
        vector_field
            Array with `Di + 1` dimensions, where `Di` is the input dimensionality.
        index_transform
            Transformation from source coordinate to array indices.
        interpolation_order
            Order of the spline interpolation used for coordinates which are not integer array indices.
        vector_axis
            Which axis of the `vector_field` contains the vector values; defaults to the last (`-1`).
        spaces
            References for source and target spaces

        Raises
        ------
        ValueError
            If index_transform's output dimensionality is not exactly one less than the vector field's number of dimensions.
        """
        xp = get_namespace(vector_field)
        sh = xp.shape(vector_field)
        in_ndim = len(sh) - 1
        tgt_ndim = sh[vector_axis]
        self.vector_field = vector_field
        if index_transform is None:
            source_ndim = in_ndim
        else:
            if in_ndim != index_transform.ndims.target:
                raise ValueError(
                    "If index_transform is given, its output dimensionality must match the vector field's shape"
                )
            source_ndim = index_transform.ndims.source
        self.index_transform = index_transform
        self.vector_axis = vector_axis % len(sh)
        self._mode = "constant"
        self._cval = np.nan
        self._order = interpolation_order
        super().__init__(NDims(source_ndim, tgt_ndim), spaces=spaces)

    def _vf_slices(self) -> Iterable[ArrayT]:
        slicing: list[slice | int] = [slice(None)] * (self.ndims.source + 1)
        for v_idx in range(self.ndims.target):
            slicing[self.vector_axis] = v_idx
            yield self.vector_field[tuple(slicing)]  # type: ignore

    def _get_vectors(self, coords: ArrayT) -> ArrayT:
        if self.index_transform is not None:
            coords = self.index_transform.apply(coords)
        else:
            coords = self._validate_coords(coords)
        xp = get_namespace(coords)
        c = xp.transpose(coords)
        out = []
        for vf in self._vf_slices():
            # TODO: in memory, could preallocate output and use scipy's output arg,
            # and then do dask differently.
            out.append(
                map_coordinates(
                    vf,
                    c,
                    order=self._order,
                    mode=self._mode,
                    cval=self._cval,
                    # can't be used with dask
                    # output=out[idx, :],
                )
            )
        return xp.transpose(xp.stack(out))

    def to_device(self, xp: ModuleType, device: str | None = None) -> Self:
        coords = xp.asarray(self.vector_field, device)
        return type(self)(coords, spaces=self.spaces)


class Coordinates(BaseVectorField[ArrayT]):
    """Look up the output coordinates in an array.

    For input coordinate `(a, b, c)` and `vector_axis=-1`,
    the output coordinate is `vector_field[a, b, c, :].

    Input coordinates outside the vector field return NaN.
    """

    def __init__(
        self,
        vector_field: ArrayT,
        index_transform: Transform[ArrayT] | None = None,
        interpolation_order: int = 3,
        vector_axis: int = -1,
        *,
        spaces: Spaces = Spaces(None, None),
    ):
        """Use the input coordinates as array indices to look up output coordinates.

        For input coordinate `(a, b, c)`, the output coordinate is `coordinates[a, b, c, :]`.

        Input coordinates outside of the `vector_field` array return `NaN` output coordinates.

        Parameters
        ----------
        vector_field
            Array with `Di + 1` dimensions, where `Di` is the input dimensionality.
        index_transform
            Transform the source coordinates into an array index
        interpolation_order
            Order of the spline interpolation used for coordinates which are not integer array indices.
        vector_axis
            Which axis of the `vector_field` contains the vector values; defaults to the last (`-1`).
        spaces
            References for source and target spaces
        """
        super().__init__(
            vector_field,
            index_transform,
            interpolation_order,
            vector_axis,
            spaces=spaces,
        )

    def apply(self, coords: ArrayT) -> ArrayT:
        return self._get_vectors(coords)


class Displacements(BaseVectorField[ArrayT]):
    """Look up a translation in an array and apply it to the input coordinates.

    For input coordinate `(a, b, c)` and `vector_axis=-1`,
    the output coordinate is `(a, b, c) + vector_field[a, b, c, :].

    Input coordinates outside the vector field return NaN.
    """

    def __init__(
        self,
        vector_field: ArrayT,
        index_transform: Transform[ArrayT] | None = None,
        interpolation_order: int = 3,
        vector_axis: int = -1,
        *,
        spaces: Spaces = Spaces(None, None),
    ):
        """
        Parameters
        ----------
        vector_field
            Array with `Di + 1` dimensions, where `Di` is the input dimensionality.
        index_transform
            Transformation from source coordinate to array indices.
        interpolation_order
            Order of the spline interpolation used for coordinates which are not integer array indices.
        vector_axis
            Which axis of the `vector_field` contains the vector values; defaults to the last (`-1`).
        spaces
            References for source and target spaces

        Raises
        ------
        ValueError
            If the index transform and vector field would change the coordinates' dimensionality,
            or the index transform's dimensionality does not match the vector field's.
        """
        super().__init__(
            vector_field,
            index_transform,
            interpolation_order,
            vector_axis,
            spaces=spaces,
        )
        if self.ndims.source != self.ndims.target:
            raise ValueError("Displacements cannot change dimensionality")

    def apply(self, coords: ArrayT) -> ArrayT:
        coords = self._validate_coords(coords)
        vecs = self._get_vectors(coords)
        return coords + vecs  # type:ignore
