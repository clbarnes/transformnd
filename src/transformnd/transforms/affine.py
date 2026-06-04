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
from ..util import as_floats, none_eq
from ..types import NDims, Spaces


class Affine(Transform[ArrayT]):
    """Affine transformation using an augmented matrix.

    The transformation matrix is stored as a NumPy array (backend-neutral).
    At apply()-time it is converted to the input coords' backend and device,
    so the transform works transparently with NumPy, JAX, PyTorch, CuPy, etc.

    Affines can be composed by matrix multiplication: `affine2 @ affine1`.
    Note that the right hand transformation is effectively applied to the coordinates first,
    so `(aff2 @ aff1).apply(coords) == (aff1 | aff2).apply(coords)`.
    """

    def __init__(
        self,
        matrix: ArrayLike,
        *,
        spaces: Spaces = Spaces(None, None),
    ):
        """
        Parameters
        ----------
        matrix : ArrayLike
            Affine transformation matrix,
            i.e. a 2D array-like with shape `(Di + 1, Do + 1)`,
            where the bottom row is all 0s except in the rightmost column, which is 1.
        spaces : Spaces
            Optional source and target spaces

        Raises
        ------
        ValueError
            Malformed matrix.
        """
        m = as_floats(matrix)
        if m.ndim != 2:
            raise ValueError("Affine matrix must be 2D")

        bottom_row = m[-1, :]
        expected = np.zeros_like(bottom_row)
        expected[-1] = 1
        if not np.allclose(bottom_row, expected):
            raise ValueError(
                f"Transformation matrix is not affine (expected bottom row {expected}, got {bottom_row})."
            )

        super().__init__(NDims(m.shape[0] - 1, m.shape[1] - 1), spaces=spaces)

        self.matrix: np.ndarray = m

        self._linear_map: np.ndarray | None = m[:-1, :-1]
        if np.allclose(
            self._linear_map, np.eye(self._linear_map.shape[0], dtype=self.matrix.dtype)
        ):
            self._linear_map = None

        self._translation: np.ndarray | None = self.matrix[:-1, -1]
        if np.allclose(np.zeros_like(self._translation), self._translation):
            self._translation = None

    def to_affine(self) -> Self | None:
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
            spaces=self.spaces.invert(),
        )

    def __matmul__(self, rhs: Affine[ArrayT]) -> Affine[ArrayT]:
        """Compose two affine transforms by matrix multiplication.

        As with affine matrices the right hand operand is effectively applied first.

        Parameters
        ----------
        rhs : Affine

        Returns
        -------
        Affine

        Raises
        ------
        ValueError
            Incompatible transforms.
        """
        if not isinstance(rhs, Affine):
            return NotImplemented
        if self.ndims.source != rhs.ndims.target:
            raise ValueError(
                "Cannot compose affine transformations of different dimensionality"
            )

        # this ordering looks wrong but this is the way affine transforms get combined;
        # the sequence transform A followed by transform B is expressed B @ A
        if not none_eq(self.spaces.source, rhs.spaces.target):
            raise ValueError("Affine transforms do not share a space")
        return Affine(
            self.matrix @ rhs.matrix,
            spaces=Spaces(rhs.spaces.source, self.spaces.target),
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
        translation: ArrayLike | None = None,
        *,
        spaces: Spaces = Spaces(None, None),
    ) -> Affine[ArrayT]:
        """Create an augmented affine matrix from a linear map,
        with an optional translation.

        Parameters
        ----------
        linear_map : ArrayLike
            Shape `(Di, Do)`
        translation : ArrayLike, optional
            Translation to add to the matrix, by default 0
        spaces : Spaces
            Optional source and target spaces

        Returns
        -------
        Affine
        """
        lin_map = as_floats(linear_map)
        if lin_map.ndim != 2:
            raise ValueError(f"Linear map must be 2D; got shape {lin_map.shape}")
        matrix = np.zeros_like(
            lin_map, shape=(lin_map.shape[0] + 1, lin_map.shape[1] + 1)
        )
        matrix[:-1, :-1] = lin_map
        matrix[-1, -1] = 1
        if translation is not None:
            t = as_floats(translation)
            if len(t) != lin_map.shape[0]:
                raise ValueError(
                    "Translation array must be the same length as linear map columns"
                )
            matrix[:-1, -1] = translation
        return cls(matrix, spaces=spaces)

    @classmethod
    def identity(
        cls,
        ndim: int,
        *,
        spaces: Spaces = Spaces(None, None),
    ) -> Affine[ArrayT]:
        """Create an identity affine transformation.

        Parameters
        ----------
        ndim : int
        spaces : Spaces
            Optional source and target spaces

        Returns
        -------
        Affine
        """
        return cls(np.eye(ndim + 1), spaces=spaces)

    @classmethod
    def translation(
        cls,
        translation: ArrayLike,
        *,
        spaces: Spaces = Spaces(None, None),
    ) -> Affine[ArrayT]:
        """Create an affine translation.

        Parameters
        ----------
        translation : ArrayLike
            D-length array of translation values.
        spaces : Spaces
            Optional source and target spaces

        Returns
        -------
        Affine
        """
        t = as_floats(translation)
        if t.ndim != 1:
            raise ValueError(f"Translation array must be 1D; got shape {t.shape}")
        m = np.eye(len(t) + 1, dtype=t.dtype)
        m[:-1, -1] = t
        return cls(m, spaces=spaces)

    @classmethod
    def scaling(
        cls,
        scale: ArrayLike,
        *,
        spaces: Spaces = Spaces(None, None),
    ) -> Affine[ArrayT]:
        """Create an affine scaling.

        Parameters
        ----------
        scale : ArrayLike
            D-length array of scaling factors.
        spaces : Spaces
            Optional source and target spaces

        Returns
        -------
        Affine
        """
        s = as_floats(scale)
        if s.ndim != 1:
            raise ValueError(f"Scale array must be 1D; got shape {s.shape}")
        return cls.from_linear_map(np.diag(s), spaces=spaces)

    @classmethod
    def reflection(
        cls,
        axis: Union[int, Container[int]],
        ndim: int,
        *,
        spaces: Spaces = Spaces(None, None),
    ) -> Affine[ArrayT]:
        """Create an affine reflection.

        Parameters
        ----------
        axis : Union[int, Container[int]]
            A single axis or multiple to reflect in.
        ndim : int
            How many dimensions to work in.
        spaces : Spaces
            Optional source and target spaces

        Returns
        -------
        Affine
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
        spaces: Spaces = Spaces(None, None),
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
        spaces : Spaces
            Optional source and target spaces

        Returns
        -------
        Affine
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
        spaces: Spaces = Spaces(None, None),
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
        spaces : Spaces
            Optional source and target spaces

        Returns
        -------
        Affine

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
        spaces: Spaces = Spaces(None, None),
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
        ndim : int, optional
            If factor is scalar, broadcast to this many dimensions, by default None
        spaces : Spaces
            Optional source and target spaces

        Returns
        -------
        Affine

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

    def is_identity(self) -> bool:
        xp = array_namespace(self.matrix)
        sh = xp.shape(self.matrix)
        if sh[0] != sh[1]:
            return False
        identity = xp.eye(sh[0], dtype=self.matrix.dtype, device=self.matrix.device)
        return xp.all(xp.equal(self.matrix, identity))
