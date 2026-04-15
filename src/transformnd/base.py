"""Base classes and wrappers for transforms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from copy import copy
from typing import Generic, Self

import numpy as np
from array_api_compat import array_namespace

from .util import (
    SpaceRef,
    TransformSignature,
    check_ndim,
    dim_intersection,
    same_or_none,
    space_str,
    window,
    SpaceTuple,
    ArrayT,
    Namespace
)


class Transform(ABC, Generic[ArrayT]):
    """Base class for transforms."""

    ndim: set[int] | None = None

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

    def _validate_coords(self, coords: ArrayT) -> ArrayT:
        """Check that dimension of coords are supported.

        Also ensure that coords is a 2D array.

        Parameters
        ----------
        coords : ArrayT
            NxD array of N D-dimensional coordinates.

        Raises
        ------
        ValueError
            If dimensions are not supported.
        """
        xp = array_namespace(coords)
        if xp.ndim(coords) != 2:
            raise ValueError("Coords must be a 2D array")
        check_ndim(xp.shape(coords)[1], self.ndim)
        return coords

    @abstractmethod
    def apply(self, coords: ArrayT) -> ArrayT:
        """Apply transformation.

        Parameters
        ----------
        coords : ArrayT
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

    def to_device(self, xp, device=None) -> Self:  # noqa: ARG002
        """Return a copy of this transform with array parameters placed on the given device.

        Useful for pre-allocating parameters on GPU before a tight apply() loop,
        avoiding per-call host-to-device transfers.

        Parameters
        ----------
        xp : array namespace
            The target array namespace (e.g. jax.numpy, torch).
        device : device object, optional
            Target device (e.g. from array_api_compat.device(array)).
            If None, uses xp's default device.

        Returns
        -------
        Transform
            A new transform instance with parameters on the target device,
            or NotImplemented if the subclass does not support device placement.
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
    """Wrapper around an arbitrary function which transforms coordinates."""

    def __init__(
        self,
        fn: TransformSignature,
        ndim: set[int] | int | None = None,
        *,
        spaces: SpaceTuple = (None, None),
    ):
        """Wrapper around an arbitrary function.

        `fn` should take and return an identically-shaped
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

    def apply(self, coords: np.ndarray) -> np.ndarray:
        self._validate_coords(coords)
        return self.fn(coords)


def _with_spaces(
    t: Transform,
    source_space: SpaceRef | None = None,
    target_space: SpaceRef | None = None,
) -> Transform:
    src_tgt = (t.source_space, t.target_space)
    src = same_or_none(src_tgt[0], source_space, default=None)
    tgt = same_or_none(src_tgt[1], target_space, default=None)
    if (src, tgt) != src_tgt:
        t = copy(t)
        t.spaces = (src, tgt)
    return t


def infer_spaces(
    transforms: Sequence[Transform], source_space=None, target_space=None
) -> list[Transform]:
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


def get_transform_list(t: Transform) -> list[Transform]:
    if isinstance(t, TransformSequence):
        return t.transforms.copy()
    else:
        return [t]


class TransformSequence(Transform[ArrayT], Sequence[Transform[ArrayT]]):
    """Chain transforms, applying one after another."""

    def __init__(
        self,
        transforms: Sequence[Transform[ArrayT]],
        *,
        spaces: SpaceTuple = (None, None),
    ) -> None:
        """Combine transforms by chaining them.

        Also checks for consistent dimensionality and space references,
        inferring if None.

        Parameters
        ----------
        transforms : List[Transform[ArrayT]]
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

        self.transforms: list[Transform] = ts

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
            spaces=(self.spaces[1], self.spaces[0]),
        )

    def apply(self, coords: ArrayT) -> ArrayT:
        for t in self.transforms:
            coords = t.apply(coords)
        return coords

    def list_spaces(self, skip_none=False) -> list[SpaceRef]:
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

    def __getitem__(self, idx: slice | int):
        if isinstance(idx, int):
            return self.transforms[idx]
        return type(self)(self.transforms[idx])
