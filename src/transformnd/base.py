"""Base classes and wrappers for transforms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from copy import copy
from typing import Self, TYPE_CHECKING
from types import ModuleType

from array_api_compat import array_namespace

from .util import (
    SpaceRef,
    same_or_none,
    space_str,
    ArrayT,
)
from itertools import pairwise

from .types import TransformSignature, Spaces, NDims

if TYPE_CHECKING:
    from .transforms import Affine


class Transform[ArrayT](ABC):
    """Base class for transforms."""

    def __init__(
        self,
        ndims: NDims,
        *,
        spaces: Spaces = Spaces(None, None),
    ):
        """
        Parameters
        ----------
        ndims
            Source and target dimensionality.
        spaces
            Optional source and target spaces
        """
        self.ndims: NDims = ndims
        self.spaces: Spaces = spaces

    def is_identity(self) -> bool:
        """Whether this is a no-op transformation."""
        return False

    def to_affine(self) -> Affine[ArrayT] | None:
        """Convert the transform into affine, if conversion is possible.

        Returns
        -------
        Affine[ArrayT] | None
            The affine transformation, if conversion is possible.
            None otherwise.
        """
        return None

    def _validate_coords(self, coords: ArrayT) -> ArrayT:
        """Check that input coordinates are of the correct shape.

        Also ensure that coords is a 2Darray.

        Parameters
        ----------
        coords
            NxD array of N D-dimensional coordinates.

        Returns
        -------
        ArrayT
            The validated coordinates.

        Raises
        ------
        ValueError
            If dimensions are not supported.
        """
        xp = array_namespace(coords)
        if xp.ndim(coords) != 2:
            raise ValueError("Coords must be a 2D array")
        dim = xp.shape(coords)[1]
        if xp.shape(coords)[1] != self.ndims.source:
            raise ValueError(
                f"Coords must have dimensionality {self.ndims.source}, got {dim}"
            )
        return coords

    @abstractmethod
    def apply(self, coords: ArrayT) -> ArrayT:
        """Apply transformation.

        Parameters
        ----------
        coords
            NxD array of N D-dimensional coordinates.

        Returns
        -------
        ArrayT
            Transformed coordinates in the same shape.
        """
        pass

    def invert(self) -> Transform | None:
        """Invert the transformation, returning `None` if not possible."""
        return None

    def __invert__(self) -> Transform:
        """Invert transformation if possible.

        Returns `NotImplemented` otherwise (will raise `NotImplementedError`).

        Returns
        -------
        Transform
            Inverted transformation.
        """
        t = self.invert()
        if t is None:
            return NotImplemented
        return t

    def to_device(self, xp: ModuleType, device: str | None = None) -> Self:  # noqa: ARG002
        """Return a copy of this transform with array parameters placed on the given device.

        Useful for pre-allocating parameters on GPU before a tight apply() loop,
        avoiding per-call host-to-device transfers.

        Parameters
        ----------
        xp
            The target array namespace (e.g. jax.numpy, torch).
        device
            Target device (e.g. from array_api_compat.device(array)).
            If None, uses xp's default device.

        Returns
        -------
        Self
            A new transform instance with parameters on the target device,
            or NotImplemented if the subclass does not support device placement.
        """
        return NotImplemented

    def __or__(self, other: Transform[ArrayT]) -> TransformSequence[ArrayT]:
        """Compose transformations into a sequence.

        If other is a TransformSequence, prepend this transform to the others.

        Parameters
        ----------
        other
            The transform to compose with.

        Returns
        -------
        TransformSequence[ArrayT]
            The composed transform sequence.
        """
        if not isinstance(other, Transform):
            return NotImplemented
        transforms = as_transform_list(self) + as_transform_list(other)
        return TransformSequence[ArrayT](
            transforms,
            spaces=Spaces(self.spaces.source, other.spaces.target),
        )

    def __ror__(self, other: Transform[ArrayT]) -> TransformSequence[ArrayT]:
        """Compose transformations into a sequence.

        If other is a TransformSequence, append this transform to the others.

        Parameters
        ----------
        other
            The transform to compose with.

        Returns
        -------
        TransformSequence[ArrayT]
            The composed transform sequence.
        """
        if not isinstance(other, Transform):
            return NotImplemented
        transforms = as_transform_list(other) + as_transform_list(self)
        return TransformSequence(
            transforms,
            spaces=Spaces(other.spaces.source, self.spaces.target),
        )

    def __str__(self) -> str:
        cls_name = type(self).__name__
        src = space_str(self.spaces.source)
        tgt = space_str(self.spaces.target)
        return f"{cls_name}[{src}->{tgt}]"


class TransformWrapper(Transform[ArrayT]):
    """Wrapper around an arbitrary function which transforms coordinates."""

    def __init__(
        self,
        fn: TransformSignature[ArrayT],
        in_ndim: int,
        out_ndim: int,
        *,
        spaces: Spaces = Spaces(None, None),
    ):
        """Wrapper around an arbitrary function.

        `fn` should take and return an identically-shaped
        NxD numpy array of N D-dimensional coordinates.

        Parameters
        ----------
        fn
            Callable.
        in_ndim
            Dimensionality of the input coordinates.
        out_ndim
            Dimensionality of the output coordinates.
        spaces
            Optional source and target spaces
        """
        super().__init__(NDims(in_ndim, out_ndim), spaces=spaces)
        self.fn = fn

    def apply(self, coords: ArrayT) -> ArrayT:
        self._validate_coords(coords)
        return self.fn(coords)


def _with_spaces(
    t: Transform[ArrayT],
    source_space: SpaceRef | None = None,
    target_space: SpaceRef | None = None,
) -> Transform[ArrayT]:
    src_tgt = (t.spaces.source, t.spaces.target)
    src = same_or_none(src_tgt[0], source_space, default=None)
    tgt = same_or_none(src_tgt[1], target_space, default=None)
    if (src, tgt) != src_tgt:
        t = copy(t)
        t.spaces = Spaces(src, tgt)
    return t


def infer_spaces(
    transforms: Sequence[Transform[ArrayT]], source_space=None, target_space=None
) -> list[Transform[ArrayT]]:
    prev_tgts = [source_space]
    next_srcs = []
    for t1, t2 in pairwise(transforms):
        prev_tgts.append(t1.spaces.target)
        next_srcs.append(t2.spaces.source)

    next_srcs.append(target_space)

    out = []
    for t, next_src, prev_tgt in zip(transforms, next_srcs, prev_tgts):
        out.append(_with_spaces(t, prev_tgt, next_src))
    return out


def as_transform_list(t: Transform[ArrayT]) -> list[Transform[ArrayT]]:
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
        spaces: Spaces = Spaces(None, None),
    ) -> None:
        """Combine transforms by chaining them.

        Also checks for consistent dimensionality and space references,
        inferring if None.

        Parameters
        ----------
        transforms :
            Items which are a TransformSequences
            will each still be treated as a single transform.
        spaces :
            Optional source and target spaces.
            Can also be inferred from the first and last transforms.

        Raises
        ------
        ValueError
            If spaces are incompatible.
        """
        ts = infer_spaces(transforms, *spaces)
        if not ts:
            raise ValueError("Empty transform sequence")

        spaces = Spaces(ts[0].spaces.source, ts[-1].spaces.target)
        ndims = NDims(ts[0].ndims.source, ts[-1].ndims.target)

        super().__init__(
            ndims,
            spaces=spaces,
        )

        self.transforms: list[Transform[ArrayT]] = ts

    def __iter__(self) -> Iterator[Transform[ArrayT]]:
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

    def invert(self) -> Transform[ArrayT] | None:
        try:
            transforms = [~t for t in reversed(self.transforms)]
        except NotImplementedError:
            return None
        return type(self)(
            transforms,
            spaces=self.spaces.invert(),
        )

    def apply(self, coords: ArrayT) -> ArrayT:
        for t in self.transforms:
            coords = t.apply(coords)
        return coords

    def to_device(self, xp: ModuleType, device: str | None = None) -> Self:
        result = copy(self)
        result.transforms = [t.to_device(xp, device) for t in self.transforms]
        return result

    def list_spaces(self, skip_none: bool = False) -> list[SpaceRef]:
        """List spaces in this transform.

        Parameters
        ----------
        skip_none
            Whether to skip undefined spaces, default False.

        Returns
        -------
        list[SpaceRef]
            The list of spaces.
        """
        spaces = [self.spaces.source] + [t.spaces.target for t in self.transforms]
        if skip_none:
            spaces = [s for s in spaces if s is not None]
        return spaces

    def split(self) -> Iterator[Transform[ArrayT]]:
        """Split the sequence where an intermediate space is known."""
        this_seq = []

        for t in self.transforms:
            if t.spaces.source is not None and t.spaces.target is not None:
                yield t
                continue

            this_seq.append(t)
            if t.spaces.target is not None:
                yield type(self)(this_seq)
                this_seq = []

    def __str__(self) -> str:
        cls_name = type(self).__name__
        spaces_str = "->".join(space_str(s) for s in self.list_spaces())
        return f"{cls_name}[{spaces_str}]"

    def __getitem__(self, idx: slice | int):
        if isinstance(idx, int):
            return self.transforms[idx]
        return type(self)(self.transforms[idx])

    def is_identity(self) -> bool:
        return all(t.is_identity() for t in self)

    def flatten(self, drop_inverse: bool = True) -> Self:
        """Flatten nested sequences."""
        from .transforms.bijection import Bijection

        out: list[Transform[ArrayT]] = []

        for t in self.transforms:
            if drop_inverse and isinstance(t, Bijection):
                t = t.forward
            if isinstance(t, TransformSequence):
                out.extend(t.flatten())
            out.append(t)
        return TransformSequence(out, spaces=self.spaces)  # type:ignore

    def simplify(self, drop_inverse: bool = True):
        """Reduce the number of transformations in this sequence if possible.

        - Compose consecutive transformations which can be expressed as affines
        - Drop trivial transforms (e.g. identity)
        - Optionally drop explicit inverse transforms
          (e.g. replace `Bijection`s with their `forward` transform)

        Also drops all internal space tuples; only the sequence's remains.

        Does not check whether transforms invert each other,
        e.g. `Translation(1) | Translation(-1)`.
        """
        from .transforms import Identity

        out: list[Transform[ArrayT]] = []
        affine = None
        for t in self.flatten(drop_inverse):
            if t.is_identity():
                continue

            new_affine = t.to_affine()

            if new_affine is None:
                if affine is not None:
                    add_to_output(affine, out)
                    affine = None
                add_to_output(t, out)
                continue

            if affine is None:
                affine = new_affine
            else:
                affine = new_affine @ affine  # type: ignore[operator]

        if affine is not None:
            add_to_output(affine, out)

        if not out:
            out.append(Identity(self.ndims.source))

        return type(self)(out, spaces=self.spaces)

    def to_affine(self) -> Affine[ArrayT] | None:
        simple = self.simplify(True)
        if len(simple) != 1:
            return None
        return simple[0].to_affine()


def add_to_output(transform: Transform, lst: list[Transform]) -> bool:
    if transform.is_identity():
        return False

    transform = copy(transform)
    transform.spaces = Spaces(None, None)
    lst.append(transform)
    return True
