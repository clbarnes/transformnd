from typing import List, Optional

import numpy as np

from .base import Transform
from .util import SpaceRef, flatten, is_square


def proj(u, v):
    return (np.inner(u, v) / np.inner(u, u)) * u


def gram_schmidt(vecs: np.ndarray):
    """ROW vecs

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
    Normals are unit vectors.
    """
    points = np.asarray(points)
    if points.ndim <= 1:
        points = np.expand_dims(points, -1)
    elif points.ndim > 2:
        raise ValueError("Points must be 2D array")

    ndim, n_points = points.shape
    n_reflections = ndim - n_points + 1

    if n_reflections <= 0:
        raise ValueError("Too many points given, must be hyperplane or lower-dim")

    point = points[:, 0]

    if n_points == 1:
        return point, list(np.eye(len(point)))

    # non-orthogonal vectors spanning provided reflective
    # transpose into row vectors for easier vectorisation
    reflective_vecs = np.diff(points, axis=1).T
    rng = np.random.default_rng(seed)
    randoms = rng.random((n_reflections, ndim))
    non_orth = np.concatenate((reflective_vecs, randoms), 0)

    basis = gram_schmidt(non_orth)
    extras = basis[-n_reflections:]
    if unitise:
        extras /= np.linalg.norm(extras, axis=1)
    return point, list(extras)


def reflect(plane_normal, plane_point, vector):
    return (
        vector
        - 2
        * (
            (np.dot(vector, plane_normal) - plane_point)
            / np.dot(plane_normal, plane_normal)
        )
        * plane_normal
    )


class Reflect(Transform):
    def __init__(
        self,
        points,
        *,
        source_space: Optional[SpaceRef] = None,
        target_space: Optional[SpaceRef] = None,
    ):
        super().__init__(source_space=source_space, target_space=target_space)
        self.ndim = {points.shape[0]}
        self.point, self.normals = get_hyperplanes(points, unitise=True)

    def __call__(self, coords: np.ndarray) -> np.ndarray:
        flat, unflatten = flatten(coords, transpose=True)
        flat -= self.point
        for n in self.normals:
            # mul->sum vectorises dot product
            # normals are unit, avoids unnecessary division by 1
            flat -= 2 * np.sum(flat * n, axis=1) * n
        flat += self.point
        return unflatten(flat)

    @classmethod
    def from_axis(
        cls,
        axis,
        origin,
        *,
        source_space: Optional[SpaceRef] = None,
        target_space: Optional[SpaceRef] = None,
    ):
        points = []
        if np.isscalar(axis):
            axis = (axis,)
        points = [origin]
        for i in range(len(origin) - len(axis)):
            if i not in axis:
                p = origin.copy()
                p[i] += 1
                points.append(p)
        return cls(
            np.array(points).T, source_space=source_space, target_space=target_space
        )
