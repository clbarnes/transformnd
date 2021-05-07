from __future__ import annotations

from abc import ABC, abstractmethod
from copy import copy
from typing import Iterator, List, Optional, Set, Union

import numpy as np

from .util import (
    SpaceRef,
    TransformSignature,
    check_ndim,
    same_or_none,
    window,
)


class Transform(ABC):
    ndim: Optional[Set[int]] = None

    def __init__(
        self,
        *,
        source_space: Optional[SpaceRef] = None,
        target_space: Optional[SpaceRef] = None,
    ):
        """Base class for transformations.

        Parameters
        ----------
        source_space : Optional[SpaceRef]
            To refer to the source space.
        target_space : Optional[SpaceRef]
            To refer to the target space.
        """
        self.source_space = source_space
        self.target_space = target_space

    def _check_ndim(self, coords):
        """Check that dimension of coords are supported.

        Parameters
        ----------
        coords : np.ndarray
            DxIxJxKx... array of coordinates where D is the dimensionality.

        Raises
        ------
        ValueError
            If dimensions are not supported.
        """
        return check_ndim(coords.shape[0], self.ndim)

    @abstractmethod
    def __call__(self, coords: np.ndarray) -> np.ndarray:
        """Apply transformation.

        Parameters
        ----------
        coords : np.ndarray
            DxIxJxKx... array of coordinates where D is the dimensionality.

        Returns
        -------
        np.ndarray
            Transformed coordinates in the same shape.
        """
        pass

    def __neg__(self) -> Transform:
        """Invert transformation if possible.

        Returns
        -------
        Transform
            Inverted transformation.
        """
        return NotImplemented

    def __add__(self, other) -> TransformSequence:
        """Compose transformations into a sequence.

        Parameters
        ----------
        other : Transform or callable
            Callables will be wrapped in a TransformWrapper.

        Returns
        -------
        TransformSequence
        """
        if isinstance(other, TransformSequence):
            transforms = copy(other.transforms)
        elif isinstance(other, Transform):
            transforms = [other]
        elif callable(other):
            transforms = [TransformWrapper(other)]
        else:
            return NotImplemented
        transforms.insert(0, self)
        return TransformSequence(
            transforms, source_space=self.source_space, target_space=other.target_space
        )

    def __radd__(self, other) -> TransformSequence:
        """Compose transformations into a sequence.

        Parameters
        ----------
        other : Transform or callable
            Callables will be wrapped in a TransformWrapper.

        Returns
        -------
        TransformSequence
        """
        if isinstance(other, TransformSequence):
            transforms = copy(other.transforms)
        elif isinstance(other, Transform):
            transforms = [other]
        elif callable(other):
            transforms = [TransformWrapper(other)]
        else:
            return NotImplemented
        transforms.append(self)
        return TransformSequence(
            transforms, source_space=other.source_space, target_space=self.target_space
        )


class TransformWrapper(Transform):
    def __init__(
        self,
        fn: TransformSignature,
        ndim: Optional[Union[Set[int], int]] = None,
        *,
        source_space: Optional[SpaceRef] = None,
        target_space: Optional[SpaceRef] = None,
    ):
        """Wrapper around an arbitrary function.

        Callable should take and return an identically-shaped
        DxIxJxKx... numpy array of coordinates,
        where D is the dimensionality.

        Parameters
        ----------
        fn : TransformSignature
            Callable.
        source_space : Optional[SpaceRef]
            Any hashable, to refer to the source space.
        target_space : Optional[SpaceRef]
            Any hashable, to refer to the target space.
        """
        super().__init__(source_space=source_space, target_space=target_space)
        self.fn = fn
        if ndim is not None:
            try:
                self.ndim = set(ndim)
            except TypeError:
                self.ndim = {ndim}

    def __call__(self, coords: np.ndarray) -> np.ndarray:
        self._check_ndim(coords)
        return self.fn(coords)


def infer_spaces(transforms: List[Transform]):
    for (t1, t2) in window(transforms, 2):
        try:
            space = same_or_none(t1.target_space, t2.source_space, default=None)
        except ValueError:
            raise ValueError("Sequence has incompatible spaces")
        t1.target_space = space
        t2.source_space = space
    return transforms


class TransformSequence(Transform):
    def __init__(
        self,
        transforms: List[Transform],
        *,
        source_space: Optional[SpaceRef] = None,
        target_space: Optional[SpaceRef] = None,
    ) -> None:
        """Combine transforms by chaining them.

        Also checks for consistent dimensionality and space references,
        inferring if None.

        Parameters
        ----------
        transforms : List[Transform]
        source_space : Optional[SpaceRef]
            Any hashable, to refer to the source space.
            Can also be inferred from the first transform.
        target_space : Optional[SpaceRef]
            Any hashable, to refer to the target space.
            Can also be inferred from the last transform.

        Raises
        ------
        ValueError
            If spaces are incompatible.
        """
        try:
            src = same_or_none(transforms[0].source_space, source_space, default=None)
            tgt = same_or_none(transforms[1].target_space, target_space, default=None)
        except ValueError:
            raise ValueError(
                "Source/target spaces are inconsistent with component transforms"
            )

        transforms[0].source_space = src
        transforms[-1].target_space = tgt

        super().__init__(
            source_space=src,
            target_space=tgt,
        )
        ts = []
        for t in transforms:
            if isinstance(t, TransformSequence):
                ts.extend(t.transforms)
            else:
                ts.append(t)

        self.transforms: List[Transform] = infer_spaces(ts)

        self.ndim = None
        for t in self.transforms:
            if t.ndim is not None:
                if self.ndim is None:
                    self.ndim = set(t.ndim)
                else:
                    self.ndim.intersection_update(t.ndim)
        if self.ndim is not None and len(self.ndim) == 0:
            raise ValueError("Transforms have incompatible dimensionalities")

    def __iter__(self) -> Iterator[Transform]:
        """Iterate through component transforms.

        Yields
        -------
        Iterator[Iterable[Transform]]
            [description]
        """
        yield from self.transforms

    def __len__(self) -> int:
        """Number of transforms.

        Returns
        -------
        int
        """
        return len(self.transforms)

    def __neg__(self) -> Transform:
        try:
            transforms = [-t for t in reversed(self.transforms)]
        except NotImplementedError:
            return NotImplemented
        return TransformSequence(
            transforms, source_space=self.target_space, target_space=self.source_space
        )

    def __call__(self, coords: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            coords = t(coords)
        return coords
