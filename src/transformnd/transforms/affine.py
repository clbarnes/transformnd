"""
Rigid transformations implemented as affine multiplications.
"""

from __future__ import annotations

import math
from typing import Container, Union, Self

import numpy as np
from numpy.typing import ArrayLike

from copy import copy

from array_api_compat import array_namespace, device as xp_device
from ..base import Transform, ArrayT
from ..util import is_square, none_eq, SpaceTuple, to_single_ndim


def arg_as_array(arg, ndim: int | None) -> np.ndarray:
    """Convert a scalar or array-like argument to a 1-D NumPy array.

    Parameters
    ----------
    arg :
        Scalar (broadcast to ndim) or array-like.
    ndim : int, optional
        Required length. If arg is scalar, used to broadcast.
    """
    if isinstance(arg, (int, float, complex)):
        if ndim is None:
            raise ValueError("Argument must be array-like or ndim must be given")
        return np.full(ndim, arg)
    arr = np.asarray(arg)
    if ndim is not None and len(arr) != ndim:
        raise ValueError("Mismatch between ndim and length of argument")
    return arr


class Affine(Transform[ArrayT]):
    """Affine transformation using an augmented matrix.

    The transformation matrix is stored as a NumPy array (backend-neutral).
    At apply()-time it is converted to the input coords' backend and device,
    so the transform works transparently with NumPy, JAX, PyTorch, CuPy, etc.
    """

    matrix: np.ndarray

    def __init__(
        self,
        matrix: ArrayLike,
        *,
        spaces: SpaceTuple = (None, None),
    ):
        """
        Affine transformation matrices' bottom row must be all zeroes except a 1 in the rightmost column.

        Matrix may have shape:

        - (D+1, D+1): an affine transformation matrix (the bottom row will be validated)
        - (D, D+1): an affine transformation matrix missing the bottom row (it will be added)
        - (D,): a vector of scales which become the first D elements of a diagonal matrix of size D+1

        Parameters
        ----------
        matrix : ArrayLike
            Affine transformation matrix. Any array-like (including JAX/PyTorch
            arrays) is accepted and converted to NumPy for storage.
        spaces : tuple[SpaceRef, SpaceRef]
            Optional source and target spaces

        Raises
        ------
        ValueError
            Malformed matrix.
        """
        super().__init__(spaces=spaces)
        m = np.asarray(matrix)

        if m.ndim == 1:
            scales = m
            m = np.eye(len(m) + 1, dtype=m.dtype)
            m[:-1, :-1] *= scales
        elif m.ndim != 2:
            raise ValueError("Transformation matrix must be 2D")

        if m.shape[1] == m.shape[0] + 1:
            base = np.eye(m.shape[1], dtype=m.dtype)
            base[:-1, :] = m
            m = base
        elif not is_square(m):
            raise ValueError("Transformation matrix must be square")
        else:
            bottom_row = m[-1, :]
            if bottom_row[-1] != 1 or not np.all(bottom_row[:-1] == 0):
                raise ValueError(
                    "Transformation matrix is not affine (bottom row must be [0,0,0,...,1])."
                )

        self.matrix = m

        self._linear_map: np.ndarray | None = m[:-1, :-1]
        if np.allclose(
            self._linear_map, np.eye(self._linear_map.shape[0], dtype=self.matrix.dtype)
        ):
            self._linear_map = None

        self._translation: np.ndarray | None = self.matrix[:-1, -1]
        if np.allclose(np.zeros_like(self._translation), self._translation):
            self._translation = None

        self.ndim = {len(self.matrix) - 1}

    def to_affine(self, ndim: int | None = None) -> Self | None:
        # check that if ndim is given, it matches expectation
        to_single_ndim(ndim, self.ndim)
        return self

    def cast_matrix(self, namespace, device) -> ArrayT:
        return namespace.asarray(self.matrix, device=device)

    def apply(self, coords: ArrayT) -> ArrayT:
        coords = self._validate_coords(coords)
        xp = array_namespace(coords)
        d = xp_device(coords)

        out = coords

        if self._linear_map is not None:
            lm = xp.asarray(self._linear_map, device=d)
            out = coords @ xp.matrix_transpose(lm)

        if self._translation is not None:
            t = xp.asarray(self._translation, device=d)
            if self._linear_map is None:
                out = coords + t
            else:
                out += t

        ## Padding and then unpadding the coords is slower, especially in C order
        # coords = xp.concatenate(
        #     [coords, xp.ones((coords.shape[0], 1), dtype=coords.dtype)],  # type: ignore[attr-defined]
        #     axis=1,
        # )
        # out: ArrayT = (coords @ m.T)[:, :-1]  # type: ignore[attr-defined]

        return out

    def invert(self) -> Self | None:
        try:
            inv = np.linalg.inv(self.matrix)
        except np.linalg.LinAlgError:
            return None

        return type(self)(
            inv,
            spaces=(self.spaces[1], self.spaces[0]),
        )

    def __matmul__(self, rhs: Affine[ArrayT]) -> Affine[ArrayT]:
        """Compose two affine transforms by matrix multiplication.

        As with affine matrices the right hand operand is effectively applied first.

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
        return Affine(
            self.matrix @ rhs.matrix,
            spaces=(self.source_space, rhs.target_space),
        )

    def to_device(self, xp, device=None) -> "Affine[ArrayT]":
        """Return a copy with the matrix placed on the given device/backend.

        Use this before a tight apply() loop to avoid per-call host-to-device
        transfers when coords live on GPU.

        Parameters
        ----------
        xp : array namespace
            Target array namespace (e.g. jax.numpy, torch).
        device : device object, optional
            Target device (e.g. from array_api_compat.device(array)).

        Returns
        -------
        Affine
            New instance with matrix on the target device.
        """
        result = copy(self)
        result.matrix = xp.asarray(self.matrix, device=device)
        return result

    @classmethod
    def from_linear_map(
        cls,
        linear_map: ArrayLike,
        translation=0,
        *,
        spaces: SpaceTuple = (None, None),
    ) -> Affine[ArrayT]:
        """Create an augmented affine matrix from a linear map,
        with an optional translation.

        Parameters
        ----------
        linear_map : ArrayLike
            Shape (D, D)
        translation : ArrayLike, optional
            Translation to add to the matrix, by default 0
        spaces : tuple[SpaceRef, SpaceRef]
            Optional source and target spaces

        Returns
        -------
        AffineTransform
        """
        lin_map = np.asarray(linear_map)
        side = len(lin_map) + 1
        matrix = np.eye(side, dtype=lin_map.dtype)
        matrix[:-1, :-1] = lin_map
        matrix[:-1, -1] = translation
        return cls(matrix, spaces=spaces)

    @classmethod
    def identity(
        cls,
        ndim: int,
        *,
        spaces: SpaceTuple = (None, None),
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
        return cls(np.eye(ndim + 1), spaces=spaces)

    @classmethod
    def translation(
        cls,
        translation: ArrayLike,
        ndim: int | None = None,
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
        m = np.eye(len(t) + 1, dtype=t.dtype)
        m[:-1, -1] = t
        return cls(m, spaces=spaces)

    @classmethod
    def scaling(
        cls,
        scale: ArrayLike,
        ndim: int | None = None,
        *,
        spaces: SpaceTuple = (None, None),
    ) -> Affine[ArrayT]:
        """Create an affine scaling.

        Parameters
        ----------
        scale : ArrayLike
            If scalar, broadcast to ndim.
        ndim : Optional[int], optional
            If scale is scalar, how many dimensions to use
        spaces : tuple[SpaceRef, SpaceRef]
            Optional source and target spaces

        Returns
        -------
        AffineTransform
        """
        s = arg_as_array(scale, ndim)
        m = np.eye(len(s) + 1, dtype=s.dtype)
        m[:-1, :-1] *= s
        return cls(m, spaces=spaces)

    @classmethod
    def reflection(
        cls,
        axis: Union[int, Container[int]],
        ndim: int,
        *,
        spaces: SpaceTuple = (None, None),
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
        if isinstance(axis, (int, np.integer)):
            axis = [axis]
        values = np.asarray([-1 if idx in axis else 1 for idx in range(ndim)])
        return cls.from_linear_map(np.diag(values.astype(float)), spaces=spaces)

    @classmethod
    def rotation2(
        cls,
        rotation: float,
        degrees=True,
        clockwise=False,
        *,
        spaces: SpaceTuple = (None, None),
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
            rotation = math.radians(rotation)
        if clockwise:
            rotation *= -1
        c, s = math.cos(rotation), math.sin(rotation)
        return cls.from_linear_map(np.array([[c, -s], [s, c]]), spaces=spaces)

    @classmethod
    def rotation3(
        cls,
        rotation: Union[float, tuple[float, float, float]],
        degrees=True,
        clockwise=False,
        order=(0, 1, 2),
        *,
        spaces: SpaceTuple = (None, None),
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
        if isinstance(rotation, (int, float)):
            r = [rotation] * 3
        else:
            r = list(rotation)

        if degrees:
            r = [math.radians(x) for x in r]
        if clockwise:
            r = [-x for x in r]

        if len(order) != 3 or set(order) != {0, 1, 2}:
            raise ValueError("Order must contain only 0, 1, 2 in any order.")

        order = list(order)
        c0, s0 = math.cos(r[0]), math.sin(r[0])
        c1, s1 = math.cos(r[1]), math.sin(r[1])
        c2, s2 = math.cos(r[2]), math.sin(r[2])

        rots = [
            np.array([[1, 0, 0], [0, c0, -s0], [0, s0, c0]]),
            np.array([[c1, 0, s1], [0, 1, 0], [-s1, 0, c1]]),
            np.array([[c2, -s2, 0], [s2, c2, 0], [0, 0, 1]]),
        ]
        rot = rots[order[0]] @ rots[order[1]] @ rots[order[2]]
        return cls.from_linear_map(rot, spaces=spaces)

    @classmethod
    def shearing(
        cls,
        factor: Union[float, np.ndarray],
        ndim: int | None = None,
        *,
        spaces: SpaceTuple = (None, None),
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
        if isinstance(factor, (int, float, complex)):
            if ndim is None:
                raise ValueError("If factor is scalar, ndim must be defined")
            s = np.full((ndim, ndim - 1), factor)
        else:
            s = np.asarray(factor)
            if s.ndim != 2 or s.shape[0] != s.shape[1] + 1:
                raise ValueError("Factor must be of shape (D, D-1)")
            ndim = s.shape[0]

        assert ndim is not None

        m = np.eye(ndim, dtype=s.dtype)
        for col_idx in range(m.shape[1]):
            it = iter(s[col_idx])
            for row_idx in range(m.shape[0] - 1):
                if m[row_idx, col_idx] == 0:
                    m[row_idx, col_idx] = next(it)
        return cls.from_linear_map(m, spaces=spaces)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Affine):
            return NotImplemented
        return np.array_equal(self.matrix, other.matrix) and self.spaces == other.spaces

    def into_affine(self, ndim: int | None = None) -> Affine[ArrayT]:
        ndim = to_single_ndim(ndim, self.ndim)
        return self

    def is_identity(self) -> bool:
        xp = array_namespace(self.matrix)
        assert self.ndim is not None
        ndim = list(self.ndim).pop()
        identity = xp.eye(ndim + 1, dtype=self.matrix.dtype, device=self.matrix.device)
        return xp.all(xp.equal(self.matrix, identity))
