from copy import copy
from typing import List, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from ..base import SpaceTuple, Transform
from ..util import is_square


def proj(u, v):
    return (np.inner(u, v) / np.inner(u, u)) * u


def gram_schmidt(vecs: np.ndarray):
    """
    https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process
    """
    if not is_square(vecs):
        raise ValueError("Wrong number of dimensions")

    out: List[np.ndarray] = []
    for v in vecs:
        b = v.copy()

        for u in out:
            b -= proj(u, v)

        out.append(b)
    return np.array(out)


def get_hyperplanes(points: np.ndarray, unitise=True, seed=None):
    """
    Reflective: point/line/.../hyperplane to be reflected around

    Returns point-normal representation.
    """
    points = np.asarray(points)
    if points.ndim <= 1:
        points = np.expand_dims(points, -1)
    elif points.ndim > 2:
        raise ValueError("Points must be 2D array")

    n_points, ndim = points.shape
    n_reflections = ndim - n_points + 1

    if n_reflections <= 0:
        raise ValueError("Too many points given, must be hyperplane or lower-dim")

    point = points[0]

    if n_points == 1:
        return point, list(np.eye(len(point)))

    # non-orthogonal vectors spanning provided reflective
    # transpose into row vectors for easier vectorisation
    reflective_vecs = np.diff(points, axis=0)
    rng = np.random.default_rng(seed)
    randoms = rng.random((n_reflections, ndim))
    non_orth = np.concatenate((reflective_vecs, randoms), 0)

    basis = gram_schmidt(non_orth)
    extras = basis[-n_reflections:]
    if unitise:
        extras /= np.linalg.norm(extras, axis=1)
    return point, list(extras)


def unitise(v):
    return v / np.linalg.norm(v)


def ensure_tuple(obj) -> Tuple:
    try:
        return tuple(obj)
    except TypeError:
        return (obj,)


class Reflect(Transform):
    def __init__(
        self,
        normals: ArrayLike,
        point=0,
        *,
        spaces: SpaceTuple = (None, None),
    ):
        """Reflection about arbitrary planes.

        Parameters
        ----------
        normals : sequence of arrays
            Normal vectors to the planes of reflection.
            Unitised internally.
        point : float or sequence of floats, optional
            Intersection point of all reflection planes
            (can be broadcast from scalar), by default 0 (i.e. the origin)
        spaces : tuple[SpaceRef, SpaceRef]
            Optional source and target spaces

        Raises
        ------
        ValueError
            Inconsistent dimensionality
        """
        super().__init__(spaces=spaces)
        normals = np.asarray(normals)
        if normals.ndim == 1:
            normals = [normals]

        n1 = normals[0]
        if not np.isscalar(point) and len(n1) != len(point):
            raise ValueError("Point and normals are not of the same dimensionality")
        self.point = point
        self.ndim = {len(n1)}
        self.normals = [unitise(n) for n in normals]
        # todo: matmul is associative, so turn this into an affine in 2/3D?

    def apply(self, coords: np.ndarray) -> np.ndarray:
        coords = self._validate_coords(coords)
        out = coords - self.point
        for n in self.normals:
            # mul->sum vectorises dot product
            # normals are unit, avoids unnecessary division by 1
            out -= 2 * np.sum(coords * n, axis=1) * n
        out += self.point
        return out

    @classmethod
    def from_points(
        cls,
        points: ArrayLike,
        *,
        spaces: SpaceTuple = (None, None),
    ):
        """Infer a single plane of reflection from a minimal number of points on it.

        Parameters
        ----------
        points :
            NxD array of N points in D dimensions. N == D
        spaces : tuple[SpaceRef, SpaceRef]
            Optional source and target spaces

        Returns
        -------
        Reflection
        """
        point, normals = get_hyperplanes(np.asarray(points), unitise=False)
        return cls(normals, point, spaces=spaces)

    @classmethod
    def from_axis(
        cls,
        axis: Union[int, Sequence[int]],
        origin: ArrayLike,
        *,
        spaces: SpaceTuple = (None, None),
    ):
        """Reflect around hyperplane(s) parallel with axes.

        Parameters
        ----------
        axis : int or sequence of int
            Index (or indices) of axes in which to reflect.
        origin : array-like
            Point around which to reflect.
        spaces : tuple[SpaceRef, SpaceRef]
            Optional source and target spaces

        Returns
        -------
        Reflection

        Raises
        ------
        ValueError
            Selected axis does not exist.
        """
        origin = np.asarray(origin)
        axis = ensure_tuple(axis)

        for a in axis:
            if a >= len(axis):
                raise ValueError(
                    "Cannot reflect in axis which does not exist (too high)"
                )

        normals = []
        for i in range(len(origin) - len(axis) + 1):
            if i not in axis:
                v = np.zeros_like(origin)
                v[i] += 1
                normals.append(v)

        return cls(normals, origin, spaces=spaces)

    def __invert__(self):
        return copy(self)
