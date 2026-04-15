"""
Rigid transformations implemented as affine multiplications.
"""

from __future__ import annotations

from typing import Container, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from array_api_compat import array_namespace
from ..base import Transform, ArrayT
from ..util import is_square, none_eq, SpaceTuple, Namespace


def arg_as_array(arg: ArrayT, ndim: Optional[int], name_space: Optional[Namespace] = None) -> ArrayT:
    if name_space is None:
        name_space = np
    if name_space.isscalar(arg):
        if ndim is None:
            raise ValueError("Argument must be array-like or ndim must be given")
        return name_space.full(ndim, arg)
    arr = name_space.asarray(arg)
    if ndim is not None and len(arr) != ndim:
        raise ValueError("Mismatch between ndim and length of argument")
    return arr


class Affine(Transform[ArrayT]):
    """Affine transformation using an augmented matrix."""
    m: ArrayT
    
    def __init__(
        self,
        matrix: ArrayLike,
        *,
        spaces: SpaceTuple = (None, None),
    ):
        """
        Matrix must have shape (D+1, D+1) or (D, D+1);
        the bottom row is assumed to be [0, 0, ..., 0, 1].

        Parameters
        ----------
        matrix : ArrayLike
            Affine transformation matrix.
        spaces : tuple[SpaceRef, SpaceRef]
            Optional source and target spaces

        Raises
        ------
        ValueError
            Malformed matrix.
        """
        super().__init__(spaces=spaces)
        xp = array_namespace(matrix)
        m = xp.asarray(matrix)

        if m.ndim != 2:
            raise ValueError("Transformation matrix must be 2D")
    
        if m.shape[1] == m.shape[0] + 1:
            base = xp.eye(m.shape[1])
            base[:-1, :] = m
            m = base
        elif not is_square(m):
            raise ValueError("Transformation matrix must be square")

        self.matrix = m
        self.ndim = {len(self.matrix) - 1}

    def apply(self, coords: ArrayT) -> ArrayT:
        coords = self._validate_coords(coords)
        # todo: replace with writing into full ones?
        xp = array_namespace(coords)
        coords = xp.asarray(coords)
        coords = xp.concatenate(
            [coords, xp.ones((coords.shape[0], 1), dtype=coords.dtype)], axis=1
        )
        out: ArrayT = (self.matrix @ coords.T).T[:, :-1]
        return out

    def __invert__(self) -> Transform:
        xp = array_namespace(self.matrix)
        return type(self)(
            xp.linalg.inv(self.matrix),
            spaces=(self.spaces[1], self.spaces[0]),
        )

    def __matmul__(self, rhs: Affine[ArrayT]) -> Affine[ArrayT]:
        """Compose two affine transforms by matrix multiplication.

        Parameters
        ----------
        rhs : AffineTransform

        Returns
        -------
        AffineTransform

        Raises
        ------
        ValueError
            Incompatible transforms.
        """
        if not isinstance(rhs, Affine):
            return NotImplemented
        if self.matrix.shape != rhs.matrix.shape:
            raise ValueError(
                "Cannot multiply affine matrices of different dimensionality"
            )
        if not none_eq(self.target_space, rhs.source_space):
            raise ValueError("Affine transforms do not share a space")
        xp = array_namespace(self.matrix)
        return Affine(
            xp.matmul(self.matrix, rhs.matrix),
            spaces=(self.source_space, rhs.target_space),
        )

    @classmethod
    def from_linear_map(
        cls,
        linear_map: ArrayT,
        translation=0,
        *,
        spaces: SpaceTuple = (None, None),
    ) -> Affine[ArrayT]:
        """Create an augmented affine matrix from a linear map,
        with an optional translation.

        Parameters
        ----------
        linear_map : ArrayT
            Shape (D, D)
        translation : ArrayT, optional
            Translation to add to the matrix, by default None
        spaces : tuple[SpaceRef, SpaceRef]
            Optional source and target spaces

        Returns
        -------
        AffineTransform
        """
        xp = array_namespace(linear_map)
        lin_map = xp.asarray(linear_map)

        side = len(lin_map) + 1
        matrix = xp.eye(side, dtype=lin_map.dtype)
        matrix[:-1, :] = lin_map
        matrix[:-1, -1] = translation

        return cls(matrix, spaces=spaces)

    @classmethod
    def identity(
        cls,
        ndim: int,
        *,
        spaces: SpaceTuple = (None, None),
        name_space: Optional[Namespace] = None,
    ) -> Affine[ArrayT]:
        """Create an identity affine transformation.

        Parameters
        ----------
        ndim : int
        spaces : tuple[SpaceRef, SpaceRef]
            Optional source and target spaces

        Returns
        -------
        AffineTransform
        """
        if name_space is None:
            name_space = np
        return cls(
            name_space.eye(ndim + 1),
            spaces=spaces,
        )

    @classmethod
    def translation(
        cls,
        translation: ArrayT,
        ndim: Optional[int] = None,
        *,
        spaces: SpaceTuple = (None, None),
    ) -> Affine[ArrayT]:
        """Create an affine translation.

        Parameters
        ----------
        translation : ArrayLike
            If scalar, broadcast to ndim.
        ndim : int, optional
            If translation is scalar, how many dims to use.
        spaces : tuple[SpaceRef, SpaceRef]
            Optional source and target spaces

        Returns
        -------
        AffineTransform
        """
        t = arg_as_array(translation, ndim)
        xp = array_namespace(t)
        m = xp.eye(len(t) + 1, dtype=t.dtype)
        m[:-1, -1] = t

        return cls(m, spaces=spaces)

    @classmethod
    def scaling(
        cls,
        scale: ArrayT,
        ndim: Optional[int] = None,
        *,
        spaces: SpaceTuple = (None, None),
    ) -> Affine:
        """Create an affine scaling.

        Parameters
        ----------
        scale : ArrayT
            If scalar, broadcast to ndim.
        ndim : Optional[int], optional
            If scale is scalar, how many dimensions to use
        spaces : tuple[SpaceRef, SpaceRef]
            Optional source and target spaces

        Returns
        -------
        AffineTransform
            [description]
        """
        s = arg_as_array(scale, ndim)
        xp = array_namespace(s)
        m = xp.eye(len(s) + 1, dtype=s.dtype)
        m[:-1, :-1] *= s

        return cls(m, spaces=spaces)

    @classmethod
    def reflection(
        cls,
        axis: Union[int, Container[int]],
        ndim: int,
        *,
        spaces: SpaceTuple = (None, None),
        name_space: Optional[Namespace] = None,
    ) -> Affine[ArrayT]:
        """Create an affine reflection.

        Parameters
        ----------
        axis : Union[int, Container[int]]
            A single axis or multiple to reflect in.
        ndim : int
            How many dimensions to work in.
        spaces : tuple[SpaceRef, SpaceRef]
            Optional source and target spaces

        Returns
        -------
        AffineTransform
        """
        if np.isscalar(axis):
            axis = [axis]
        values = [-1 if idx in axis else 1 for idx in range(ndim)]  # type:ignore
        if name_space is None:
            name_space = np
        values = name_space.asarray(values)
        m = name_space.eye(ndim + 1)
        m[:-1, :-1] *= values

        return cls(m, spaces=spaces)

    @classmethod
    def rotation2(
        cls,
        rotation: float,
        degrees=True,
        clockwise=False,
        *,
        spaces: SpaceTuple = (None, None),
        name_space: Optional[Namespace] = None,
    ) -> Affine[ArrayT]:
        """Create a 2D affine rotation.

        Parameters
        ----------
        rotation : float
            Angle to rotate.
        degrees : bool, optional
            Whether rotation is in degrees (rather than radians), by default True
        clockwise : bool, optional
            Whether rotation is clockwise, by default False
        spaces : tuple[SpaceRef, SpaceRef]
            Optional source and target spaces

        Returns
        -------
        AffineTransform
        """
        if degrees:
            rotation = np.deg2rad(rotation)
        if clockwise:
            rotation *= -1

        vals = [
            [np.cos(rotation), -np.sin(rotation)],
            [np.sin(rotation), np.cos(rotation)],
        ]
        m = np.eye(3)
        m[:-1, :-1] = vals
        if name_space is not None:
            m = name_space.asarray(m)
        return cls(m, spaces=spaces)

    @classmethod
    def rotation3(
        cls,
        rotation: Union[float, Tuple[float, float, float]],
        degrees=True,
        clockwise=False,
        order=(0, 1, 2),
        *,
        spaces: SpaceTuple = (None, None),
        name_space: Optional[Namespace] = None,
    ) -> Affine[ArrayT]:
        """Create a 3D affine rotation.

        Parameters
        ----------
        rotation : Union[float, Tuple[float, float, float]]
            Either a single rotation for all axes, or 1 for each.
        degrees : bool, optional
            Whether rotation is in degrees (rather than radians), by default True
        clockwise : bool, optional
            Whether rotation is clockwise, by default False
        order : tuple, optional
            What order to apply the rotations, by default (0, 1, 2)
        spaces : tuple[SpaceRef, SpaceRef]
            Optional source and target spaces

        Returns
        -------
        AffineTransform

        Raises
        ------
        ValueError
            Incompatible order.
        """
        if np.isscalar(rotation):
            r = np.array([rotation] * 3)
        else:
            r = np.asarray(rotation)

        if degrees:
            r = np.deg2rad(r)
        if clockwise:
            r *= -1

        if len(order) != 3 or set(order) != {0, 1, 2}:
            raise ValueError("Order must contain only 0, 1, 2 in any order.")

        order = list(order)

        rots = [
            np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(r[0]), -np.sin(r[0])],
                    [0, np.sin(r[0]), np.cos(r[0])],
                ]
            ),
            np.array(
                [
                    [np.cos(r[1]), 0, np.sin(r[1])],
                    [0, 1, 0],
                    [-np.sin(r[1]), 0, np.cos(r[1])],
                ]
            ),
            np.array(
                [
                    [np.cos(r[2]), -np.sin(r[2]), 0],
                    [np.sin(r[2]), np.cos(r[2]), 0],
                    [0, 0, 1],
                ]
            ),
        ]

        rot = rots[order.pop(0)]
        rot @= rots[order.pop(0)]
        rot @= rots[order.pop(0)]

        m = np.eye(4)
        m[:-1, :-1] = rot

        if name_space is not None:
            m = name_space.asarray(m)
        return cls(m, spaces=spaces)

    @classmethod
    def shearing(
        cls,
        factor: Union[float, Namespace],
        ndim: Optional[int] = None,
        *,
        spaces: SpaceTuple = (None, None),
        name_space: Optional[Namespace] = None,
    ) -> Affine[ArrayT]:
        """Create an affine shear.

        `factor` can be a scalar to broadcast to all dimensions,
        or a D-length list of D-1 lists.
        The first inner list contains the shear factors in the first dimension
        for all *but* the first dimension.
        The second inner list contains the shear factors in the second dimension
        for all the *but* the second dimension, etc.

        Parameters
        ----------
        factor : Union[float, np.ndarray]
            Shear scale factors; see above for more details.
        ndim : Optional[int], optional
            If factor is scalar, broadcast to this many dimensions, by default None
        spaces : tuple[SpaceRef, SpaceRef]
            Optional source and target spaces

        Returns
        -------
        AffineTransform

        Raises
        ------
        ValueError
            Incompatible factor.
        """
        if np.isscalar(factor):
            if ndim is None:
                raise ValueError("If factor is scalar, ndim must be defined")
            s = np.full((ndim, ndim - 1), factor)
            name_space = array_namespace(s)
        else:
            name_space = array_namespace(factor)
            s = name_space.asarray(factor)
            if s.shape[0] != s.shape[1] + 1:
                raise ValueError("Factor must be of shape (D, D-1)")
            ndim = s.shape[0]

        # needed for type checking
        assert ndim is not None

        m = name_space.eye(ndim, dtype=s.dtype)
        for col_idx in range(m.shape[1]):
            it = iter(factor[col_idx])  # type: ignore
            for row_idx in range(m.shape[0] - 1):
                if m[row_idx, col_idx] == 0:
                    m[row_idx, col_idx] = next(it)
        return cls(m, spaces=spaces)
