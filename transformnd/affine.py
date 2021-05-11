from __future__ import annotations

from typing import Container, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from .base import Transform
from .util import SpaceRef, is_square, none_eq


def arg_as_array(arg: ArrayLike, ndim: Optional[int]):
    if np.isscalar(arg):
        if ndim is None:
            raise ValueError("Argument must be array-like or ndim must be given")
        return np.full(ndim, arg)
    arr = np.asarray(arg)
    if ndim is not None and len(arr) != ndim:
        raise ValueError("Mismatch between ndim and length of argument")
    return arr


class AffineTransform(Transform):
    def __init__(
        self,
        matrix: ArrayLike,
        *,
        source_space: Optional[SpaceRef] = None,
        target_space: Optional[SpaceRef] = None,
    ):
        """Affine transformation using an augmented matrix.

        Matrix must have shape (D+1, D+1) or (D, D+1);
        the bottom row is assumed to be [0, 0, ..., 0, 1].

        Parameters
        ----------
        matrix : ArrayLike
            Affine transformation matrix.
        source_space : Optional[SpaceRef], optional
        target_space : Optional[SpaceRef], optional

        Raises
        ------
        ValueError
            Malformed matrix.
        """
        super().__init__(source_space=source_space, target_space=target_space)
        m = np.asarray(matrix)

        if m.ndim != 2:
            raise ValueError("Transformation matrix must be 2D")

        if m.shape[1] == m.shape[0] + 1:
            base = np.eye(m.shape[1])
            base[:-1, :] = m
            m = base
        elif not is_square(m):
            raise ValueError("Transformation matrix must be square")

        self.matrix = m
        self.ndim = {len(self.matrix) - 1}

    def __call__(self, coords: np.ndarray) -> np.ndarray:
        coords = self._check_ndim(coords)
        # todo: replace with writing into full ones?
        coords = np.concatenate(
            [coords, np.ones((coords.shape[0], 1), dtype=coords.dtype)], axis=1
        )
        return (self.matrix @ coords.T).T[:, :-1]

    def __neg__(self) -> Transform:
        return type(self)(
            np.linalg.inv(self.matrix),
            source_space=self.target_space,
            target_space=self.source_space,
        )

    def __matmul__(self, rhs: AffineTransform) -> AffineTransform:
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
        if not isinstance(rhs, AffineTransform):
            return NotImplemented
        if self.matrix.shape != rhs.matrix.shape:
            raise ValueError(
                "Cannot multiply affine matrices of different dimensionality"
            )
        if not none_eq(self.target_space, rhs.source_space):
            raise ValueError("Affine transforms do not share a space")
        return AffineTransform(
            self.matrix @ rhs.matrix,
            source_space=self.source_space,
            target_space=rhs.target_space,
        )

    @classmethod
    def from_linear_map(
        cls,
        linear_map: ArrayLike,
        translation=0,
        *,
        source_space: Optional[SpaceRef] = None,
        target_space: Optional[SpaceRef] = None,
    ):
        """Create an augmented affine matrix from a linear map,
        with an optional translation.

        Parameters
        ----------
        linear_map : ArrayLike
            Shape (D, D)
        translation : ArrayLike, optional
            Translation to add to the matrix, by default None
        source_space : Optional[SpaceRef], optional
        target_space : Optional[SpaceRef], optional

        Returns
        -------
        [type]
            [description]
        """
        lin_map = np.asarray(linear_map)

        side = len(lin_map) + 1
        matrix = np.eye(shape=(side, side), dtype=lin_map.dtype)
        matrix[:-1, :] = lin_map
        matrix[:-1, -1] = translation

        return cls(matrix, source_space=source_space, target_space=target_space)

    @classmethod
    def identity(
        cls,
        ndim: int,
        *,
        source_space: Optional[SpaceRef] = None,
        target_space: Optional[SpaceRef] = None,
    ) -> AffineTransform:
        """Create an identity affine transformation.

        Parameters
        ----------
        ndim : int
        source_space : Optional[SpaceRef], optional
        target_space : Optional[SpaceRef], optional

        Returns
        -------
        [type]
            [description]
        """
        return cls(
            np.eye(ndim + 1), source_space=source_space, target_space=target_space
        )

    @classmethod
    def translation(
        cls,
        translation: ArrayLike,
        ndim: Optional[int] = None,
        *,
        source_space: Optional[SpaceRef] = None,
        target_space: Optional[SpaceRef] = None,
    ) -> AffineTransform:
        """Create an affine translation.

        Parameters
        ----------
        translation : ArrayLike
            If scalar, broadcast to ndim.
        ndim : int, optional
            If translation is scalar, how many dims to use.
        source_space : Optional[SpaceRef], optional
        target_space : Optional[SpaceRef], optional

        Returns
        -------
        AffineTransform
        """
        t = arg_as_array(translation, ndim)
        m = np.eye(len(t) + 1, dtype=t.dtype)
        m[:-1, -1] = t

        return cls(m, source_space=source_space, target_space=target_space)

    @classmethod
    def scaling(
        cls,
        scale: ArrayLike,
        ndim: Optional[int] = None,
        *,
        source_space: Optional[SpaceRef] = None,
        target_space: Optional[SpaceRef] = None,
    ) -> AffineTransform:
        """Create an affine scaling.

        Parameters
        ----------
        scale : ArrayLike
            If scalar, broadcast to ndim.
        ndim : Optional[int], optional
            If scale is scalar, how many dimensions to use
        source_space : Optional[SpaceRef], optional
        target_space : Optional[SpaceRef], optional

        Returns
        -------
        AffineTransform
            [description]
        """
        s = arg_as_array(scale, ndim)
        m = np.eye(len(s) + 1, dtype=s.dtype)
        m[:-1, :-1] *= s

        return cls(m, source_space=source_space, target_space=target_space)

    @classmethod
    def reflection(
        cls,
        axis: Union[int, Container[int]],
        ndim: int,
        *,
        source_space: Optional[SpaceRef] = None,
        target_space: Optional[SpaceRef] = None,
    ) -> AffineTransform:
        """Create an affine reflection.

        Parameters
        ----------
        axis : Union[int, Container[int]]
            A single axis or multiple to reflect in.
        ndim : int
            How many dimensions to work in.
        source_space : Optional[SpaceRef], optional
        target_space : Optional[SpaceRef], optional

        Returns
        -------
        AffineTransform
        """
        if np.isscalar(axis):
            axis = [axis]
        values = [-1 if idx in axis else 1 for idx in range(ndim)]
        m = np.eye(ndim + 1)
        m[:-1, :-1] *= values

        return cls(m, source_space=source_space, target_space=target_space)

    @classmethod
    def rotation2(
        cls,
        rotation: float,
        degrees=True,
        clockwise=False,
        *,
        source_space: Optional[SpaceRef] = None,
        target_space: Optional[SpaceRef] = None,
    ) -> AffineTransform:
        """Create a 2D affine rotation.

        Parameters
        ----------
        rotation : float
            Angle to rotate.
        degrees : bool, optional
            Whether rotation is in degrees (rather than radians), by default True
        clockwise : bool, optional
            Whether rotation is clockwise, by default False
        source_space : Optional[SpaceRef], optional
        target_space : Optional[SpaceRef], optional

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

        return cls(m, source_space=source_space, target_space=target_space)

    @classmethod
    def rotation3(
        cls,
        rotation: Union[float, Tuple[float, float, float]],
        degrees=True,
        clockwise=False,
        order=(0, 1, 2),
        *,
        source_space: Optional[SpaceRef] = None,
        target_space: Optional[SpaceRef] = None,
    ) -> AffineTransform:
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
        source_space : Optional[SpaceRef], optional
        target_space : Optional[SpaceRef], optional

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

        return cls(m, source_space=source_space, target_space=target_space)

    @classmethod
    def shearing(
        cls,
        factor: Union[float, np.ndarray],
        ndim: Optional[int] = None,
        *,
        source_space: Optional[SpaceRef] = None,
        target_space: Optional[SpaceRef] = None,
    ) -> AffineTransform:
        """Create an affine shear.

        ``factor`` can be a scalar to broadcast to all dimensions,
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
        source_space : Optional[SpaceRef], optional
        target_space : Optional[SpaceRef], optional

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
        else:
            s = np.asarray(factor)
            if s.shape[0] != s.shape[1] + 1:
                raise ValueError("Factor must be of shape (D, D-1)")
            ndim = s.shape[0]

        m = np.eye(ndim, dtype=s.dtype)
        for col_idx in range(m.shape[1]):
            it = iter(factor[col_idx])
            for row_idx in range(m.shape[0] - 1):
                if m[row_idx, col_idx] == 0:
                    m[row_idx, col_idx] = next(it)

        return cls(m, source_space=source_space, target_space=target_space)
