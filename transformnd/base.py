"""Base classes and wrappers for transforms."""
from __future__ import annotations

from abc import ABC, abstractmethod
from copy import copy
from typing import Iterator, List, Optional, Set, Tuple, Union

import numpy as np

from .util import (
    SpaceRef,
    TransformSignature,
    check_ndim,
    dim_intersection,
    same_or_none,
    space_str,
    window,
)

SpaceTuple = Tuple[Optional[SpaceRef], Optional[SpaceRef]]


class Transform(ABC):
    ndim: Optional[Set[int]] = None

    def __init__(
        self,
        *,
        spaces: SpaceTuple = (None, None),
    ):
        """Base class for transformations.

        Parameters
        ----------
        spaces : tuple[SpaceRef, SpaceRef]
            Optional source and target spaces
        """
        self.spaces = spaces

    @property
    def source_space(self):
        return self.spaces[0]

    @property
    def target_space(self):
        return self.spaces[1]

    def _validate_coords(self, coords) -> np.ndarray:
        """Check that dimension of coords are supported.

        Also ensure that coords is a 2D numpy array.

        Parameters
        ----------
        coords : np.ndarray
            NxD array of N D-dimensional coordinates.

        Raises
        ------
        ValueError
            If dimensions are not supported.
        """
        coords = np.asarray(coords)
        if coords.ndim != 2:
            raise ValueError("Coords must be a 2D array")
        check_ndim(coords.shape[1], self.ndim)
        return coords

    @abstractmethod
    def __call__(self, coords: np.ndarray) -> np.ndarray:
        """Apply transformation.

        Parameters
        ----------
        coords : np.ndarray
            NxD array of N D-dimensional coordinates.

        Returns
        -------
        np.ndarray
            Transformed coordinates in the same shape.
        """
        pass

    def __invert__(self) -> Transform:
        """Invert transformation if possible.

        Returns
        -------
        Transform
            Inverted transformation.
        """
        return NotImplemented

    def __or__(self, other) -> TransformSequence:
        """Compose transformations into a sequence.

        If other is a TransformSequence, prepend this transform to the others.

        Parameters
        ----------
        other : Transform

        Returns
        -------
        TransformSequence
        """
        if not isinstance(other, Transform):
            return NotImplemented
        transforms = get_transform_list(self) + get_transform_list(other)
        return TransformSequence(
            transforms,
            spaces=(self.source_space, other.target_space),
        )

    def __ror__(self, other) -> TransformSequence:
        """Compose transformations into a sequence.

        If other is a TransformSequence, append this transform to the others.

        Parameters
        ----------
        other : Transform

        Returns
        -------
        TransformSequence
        """
        if not isinstance(other, Transform):
            return NotImplemented
        transforms = get_transform_list(other) + get_transform_list(self)
        return TransformSequence(
            transforms,
            spaces=(other.source_space, self.target_space),
        )

    def __str__(self) -> str:
        cls_name = type(self).__name__
        src = space_str(self.source_space)
        tgt = space_str(self.target_space)
        return f"{cls_name}[{src}->{tgt}]"


class TransformWrapper(Transform):
    def __init__(
        self,
        fn: TransformSignature,
        ndim: Optional[Union[Set[int], int]] = None,
        *,
        spaces: SpaceTuple = (None, None),
    ):
        """Wrapper around an arbitrary function.

        Callable should take and return an identically-shaped
        NxD numpy array of N D-dimensional coordinates.

        Parameters
        ----------
        fn : TransformSignature
            Callable.
        spaces : tuple[SpaceRef, SpaceRef]
            Optional source and target spaces
        """
        super().__init__(spaces=spaces)
        self.fn = fn
        if ndim is not None:
            if isinstance(ndim, int):
                self.ndim = {ndim}
            else:
                self.ndim = set(ndim)

    def __call__(self, coords: np.ndarray) -> np.ndarray:
        self._validate_coords(coords)
        return self.fn(coords)


def _with_spaces(t: Transform, source_space=None, target_space=None):
    src_tgt = (t.source_space, t.target_space)
    src = same_or_none(src_tgt[0], source_space, default=None)
    tgt = same_or_none(src_tgt[1], target_space, default=None)
    if (src, tgt) != src_tgt:
        t = copy(t)
        t.spaces = (src, tgt)
    return t


def infer_spaces(
    transforms: List[Transform], source_space=None, target_space=None
) -> List[Transform]:
    prev_tgts = [source_space]
    next_srcs = []
    for t1, t2 in window(transforms, 2):
        prev_tgts.append(t1.target_space)
        next_srcs.append(t2.source_space)

    next_srcs.append(target_space)

    out = []
    for t, next_src, prev_tgt in zip(transforms, next_srcs, prev_tgts):
        out.append(_with_spaces(t, prev_tgt, next_src))
    return out


def get_transform_list(t: Transform) -> List[Transform]:
    if isinstance(t, TransformSequence):
        return t.transforms.copy()
    else:
        return [t]


class TransformSequence(Transform):
    def __init__(
        self,
        transforms: List[Transform],
        *,
        spaces: SpaceTuple = (None, None),
    ) -> None:
        """Combine transforms by chaining them.

        Also checks for consistent dimensionality and space references,
        inferring if None.

        Parameters
        ----------
        transforms : List[Transform]
            Items which are a TransformSequences
            will each still be treated as a single transform.
        spaces : tuple[SpaceRef, SpaceRef]
            Optional source and target spaces.
            Can also be inferred from the first and last transforms.

        Raises
        ------
        ValueError
            If spaces are incompatible.
        """
        ts = infer_spaces(transforms, *spaces)

        super().__init__(
            spaces=(
                ts[0].source_space,
                ts[-1].target_space,
            ),
        )

        self.transforms: List[Transform] = ts

        self.ndim = None
        for t in self.transforms:
            self.ndim = dim_intersection(self.ndim, t.ndim)

        if self.ndim is not None and len(self.ndim) == 0:
            raise ValueError("Transforms have incompatible dimensionalities")

    def __iter__(self) -> Iterator[Transform]:
        """Iterate through component transforms.

        Yields
        -------
        Transform
        """
        yield from self.transforms

    def __len__(self) -> int:
        """Number of transforms.

        Returns
        -------
        int
        """
        return len(self.transforms)

    def __invert__(self) -> Transform:
        try:
            transforms = [~t for t in reversed(self.transforms)]
        except NotImplementedError:
            return NotImplemented
        return type(self)(
            transforms,
            spaces=self.spaces[::-1],
        )

    def __call__(self, coords: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            coords = t(coords)
        return coords

    def list_spaces(self, skip_none=False) -> List[SpaceRef]:
        """List spaces in this transform.

        Parameters
        ----------
        skip_none : bool, optional
            Whether to skip undefined spaces, default False.

        Returns
        -------
        List[SpaceRef]
        """
        spaces = [self.source_space] + [t.target_space for t in self.transforms]
        if skip_none:
            spaces = [s for s in spaces if s is not None]
        return spaces

    def __str__(self) -> str:
        cls_name = type(self).__name__
        spaces_str = "->".join(space_str(s) for s in self.list_spaces())
        return f"{cls_name}[{spaces_str}]"

    def __getitem__(self, idx: Union[slice, int]):
        if isinstance(idx, int):
            return self.transforms[idx]
        return type(self)(self.transforms[idx])
