"""Base classes and wrappers for transforms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from copy import copy
from typing import Self, TYPE_CHECKING

from array_api_compat import array_namespace

from .util import (
    SpaceRef,
    TransformSignature,
    check_ndim,
    dim_intersection,
    invert_spaces,
    same_or_none,
    space_str,
    to_single_ndim,
    window,
    SpaceTuple,
    ArrayT,
)

if TYPE_CHECKING:
    from .transforms import Affine


class Transform[ArrayT](ABC):
    """Base class for transforms."""

    ndim: set[int] | None = None

    def __init__(
        self,
        *,
        spaces: SpaceTuple = (None, None),
    ):
        """
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

    def is_identity(self) -> bool:
        """Whether this is a no-op transformation."""
        return False

    def to_affine(self, ndim: int | None = None) -> Affine[ArrayT] | None:
        """Convert the transform into affine, if conversion is possible.

        Parameters
        ----------
        dim: int, optional
            Total number of dimensions; If None, dim is set equal to self.ndim.

        Returns
        -------
        Transform | None:
            The affine transformation, if conversion is possible.
            None otherwise.
        """
        return None

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

    def invert(self) -> Transform | None:
        """Invert the transformation, returning `None` if not possible."""
        if self.is_identity():
            return copy(self)
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

    def __or__(self, other: Transform[ArrayT]) -> TransformSequence[ArrayT]:
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
        return TransformSequence[ArrayT](
            transforms,
            spaces=(self.source_space, other.target_space),
        )

    def __ror__(self, other: Transform[ArrayT]) -> TransformSequence[ArrayT]:
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


class TransformWrapper(Transform[ArrayT]):
    """Wrapper around an arbitrary function which transforms coordinates."""

    def __init__(
        self,
        fn: TransformSignature[ArrayT],
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

    def apply(self, coords: ArrayT) -> ArrayT:
        self._validate_coords(coords)
        return self.fn(coords)


def _with_spaces(
    t: Transform[ArrayT],
    source_space: SpaceRef | None = None,
    target_space: SpaceRef | None = None,
) -> Transform[ArrayT]:
    src_tgt = (t.source_space, t.target_space)
    src = same_or_none(src_tgt[0], source_space, default=None)
    tgt = same_or_none(src_tgt[1], target_space, default=None)
    if (src, tgt) != src_tgt:
        t = copy(t)
        t.spaces = (src, tgt)
    return t


def infer_spaces(
    transforms: Sequence[Transform[ArrayT]], source_space=None, target_space=None
) -> list[Transform[ArrayT]]:
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


def get_transform_list(t: Transform[ArrayT]) -> list[Transform[ArrayT]]:
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

        self.transforms: list[Transform[ArrayT]] = ts

        self.ndim = None
        for t in self.transforms:
            self.ndim = dim_intersection(self.ndim, t.ndim)

        if self.ndim is not None and len(self.ndim) == 0:
            raise ValueError("Transforms have incompatible dimensionalities")

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
            spaces=invert_spaces(self.spaces),
        )

    def apply(self, coords: ArrayT) -> ArrayT:
        for t in self.transforms:
            coords = t.apply(coords)
        return coords

    def to_device(self, xp, device=None) -> Self:
        result = copy(self)
        result.transforms = [t.to_device(xp, device) for t in self.transforms]
        return result

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

    def is_identity(self) -> bool:
        return all(t.is_identity() for t in self)

    def simplify(self, ndim: int | None = None, drop_inverse: bool = False):
        """Reduce the number of transformations in this sequence if possible.

        - Compose consecutive transformations which can be expressed as affines
        - Drop trivial transforms (e.g. identity)
        - Optionally drop explicit inverse transforms
          (e.g. replace `Bijection`s with their `forward` transform)

        Also drops all internal space tuples; only the sequence's remains.

        Does not check whether transforms invert each other,
        e.g. `Translation(1) | Translation(-1)`.
        """
        from .transforms.bijection import Bijection

        ndim = to_single_ndim(ndim, self.ndim)
        out: list[Transform[ArrayT]] = []
        affine = None
        for t in self.transforms:
            if drop_inverse and isinstance(t, Bijection):
                t = t.forward

            new_affine = t.to_affine(ndim)

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

        return type(self)(out)


def add_to_output(transform: Transform, lst: list[Transform]) -> bool:
    if transform.is_identity():
        return False

    transform = copy(transform)
    transform.spaces = (None, None)
    lst.append(transform)
    return True
