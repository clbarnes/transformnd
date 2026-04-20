"""Utilities used elsewhere in the package."""

from collections import deque
from collections.abc import Callable, Hashable, Iterable, Iterator
from typing import Any

# required for TypeVar(default=) argument
from typing_extensions import TypeVar

import numpy as np
from array_api_compat import array_namespace

UNSPECIFIED_SPACE_NAME = "???"

ArrayT = TypeVar("ArrayT", default=np.ndarray)

TransformSignature = Callable[[ArrayT], ArrayT]
"""Type annotation of a function which can be used as a transform."""

SpaceRef = Hashable
"""Type annotation of things which can be used to refer to spaces"""

SpaceTuple = tuple[SpaceRef | None, SpaceRef | None]


def none_eq(a: Any | None, b: Any | None) -> bool:
    """Check whether either is None or both are equal.

    Parameters
    ----------
    a : Optional[Any]
    b : Optional[Any]

    Returns
    -------
    bool
    """
    return a == b or a is None or b is None


class NoDefault:
    pass


NO_DEFAULT = NoDefault()


def chain_or[T](*args: T | None, default: NoDefault | T = NO_DEFAULT) -> T:
    """Return the first of *args which is not None.

    Can either error or return a default if there are no non-None args.

    Parameters
    ----------
    default : any, optional
        By default, raises a ValueError if *args are exhausted.
        If given, returns the given value instead.

    Returns
    -------
    Any
        One of the given args, or the default.

    Raises
    ------
    ValueError
        If `default` is not given and there are no non-None args.
    """
    for arg in args:
        if arg is not None:
            return arg
    if isinstance(default, NoDefault):
        raise ValueError("No non-None arguments")
    return default


def same_or_none[T](*args: T, default: NoDefault | T = NO_DEFAULT) -> T:
    """Check args are the same or None.

    If so, return the non-None value.
    Otherwise, raise a ValueError.

    Parameters
    ----------
    default : Any, optional
        If given, return this instead of an error
        if all *args are None.

    Returns
    -------
    Any
        The non-None arg value.

    Raises
    ------
    ValueError
        Arguments are not None, or the same.
    ValueError
        No non-None arguments found and no default given.
    """
    prev = None

    for arg in args:
        if arg is None:
            continue
        if prev is not None and prev != arg:
            raise ValueError("Arguments are not None or the same")
        prev = arg

    if prev is None:
        if isinstance(default, NoDefault):
            raise ValueError("No non-None arguments found")
        return default

    return prev


def window[T](iterable: Iterable[T], length: int) -> Iterator[tuple[T, ...]]:
    """Sliding window over iterable.

    e.g. `(it[0], it[1]), (it[1], it[2]), (it[2], it[3]), ...`

    Parameters
    ----------
    iterable : Iterable
    length : int
        Length of windows to return.

    Yields
    -------
    Tuple[Any, ...]
    """
    it = iter(iterable)
    q: deque[Any] = deque(maxlen=length)
    for _ in range(length):
        try:
            item = next(it)
        except StopIteration:
            return
        q.append(item)
    yield tuple(q)
    for item in it:
        q.append(item)
        yield tuple(q)


def check_ndim(given_ndim: int, supported_ndim: set[int] | None) -> None:
    """Raise a ValueError if dimensionality is unsupported.

    Parameters
    ----------
    given_ndim : int
        The dimensionality to check.
    supported_ndim : Optional[Set[int]]
        Which dimensions are supported.
        If None, the check passes.

    Raises
    ------
    ValueError
        If supported dimensions are defined and given_ndim is not in them.
    """
    if supported_ndim is not None and given_ndim not in supported_ndim:
        raise ValueError(
            f"Transform supported for {format_dims(supported_ndim)}, not {given_ndim}"
        )


def format_dims(supported: set[int] | None) -> str:
    """Format supported dimensions for e.g. error messages.

    Parameters
    ----------
    supported : Iterable[int]
        The supported dimensions.

    Returns
    -------
    str
        e.g. "2D/3D/4D"
    """
    if supported is None:
        return "ND"
    if not len(supported):
        return "nullD"
    return "/".join(f"{d}D" for d in sorted(supported))


def space_str(space: SpaceRef | None, default=UNSPECIFIED_SPACE_NAME) -> str:
    """Get a string representation of a space reference, with a default for None."""
    if space is None:
        return default
    else:
        return str(space)


def is_square(arr: ArrayT) -> bool:
    """Check whether an array is 2D and has the same number of rows as columns"""
    xp = array_namespace(arr)
    try:
        s1, s2 = xp.shape(arr)
    except ValueError as e:
        if "values to unpack" in str(e):
            return False
        raise e

    return s1 == s2


def dim_intersection(
    dims1: set[int] | None, dims2: set[int] | None, error_on_empty: bool = False
) -> set[int] | None:
    """Find the intersection between two sets of constraints.

    None means no constraints.
    If `error_on_empty` is truthy and there is no intersection, raise an error.
    """
    if dims1 is None:
        out = dims2
    elif dims2 is None:
        out = dims1
    else:
        out = dims1.intersection(dims2)
    if error_on_empty and out is not None and len(out) == 0:
        raise ValueError(f"incompatible dimensions: {dims1} ∩ {dims2}")
    return out


def invert_spaces(spaces: SpaceTuple) -> SpaceTuple:
    """Invert the given (source, target) space tuple."""
    return (spaces[1], spaces[0])


def to_single_ndim(ndim: None | int = None, ndims: None | set[int] = None) -> int:
    """Select a single ndim from the given options.

    Error if a single dimension cannot be selected;
    i.e. both are None or there is a conflict.

    Useful when converting a transformation with multi-dimensionality support
    (e.g. a scalar translation) into one with single-dimensionality support
    (e.g. an affine).
    """
    if ndim is None:
        if ndims is None:
            raise ValueError("no ndims specified")
        if len(ndims) != 1:
            raise ValueError(f"needs exactly one ndim, got {ndims}")
        return list(ndims).pop()

    if ndims is None or ndim in ndims:
        return ndim

    raise ValueError(f"dimensionality conflict: {ndim} not in {ndims}")
