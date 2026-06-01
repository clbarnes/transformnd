"""Utilities used elsewhere in the package."""

from typing import Any
from array_api_compat import array_namespace
from numpy.typing import ArrayLike
import numpy as np

from .types import SpaceRef, ArrayT
from .constants import UNSPECIFIED_SPACE_NAME


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


NO_DEFAULT = object()


def chain_or(*args: Any | None, default=NO_DEFAULT) -> Any:
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
    if default is NO_DEFAULT:
        raise ValueError("No non-None arguments")
    return default


def same_or_none(*args: Any, default=NO_DEFAULT) -> Any:
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
        if default is NO_DEFAULT:
            raise ValueError("No non-None arguments found")
        return default

    return prev


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


def space_str(space: SpaceRef | None) -> str:
    if space is None:
        return UNSPECIFIED_SPACE_NAME
    else:
        return str(space)


def is_square(arr: ArrayT) -> bool:
    """Check whether an array is 2D and has the same number of rows as columns"""
    xp = array_namespace(arr)
    ndim, shape = xp.ndim(arr), xp.shape(arr)
    return ndim == 2 and shape[0] == shape[1]


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


def as_floats(arr: ArrayLike):
    """Get array-like as a numpy array, casting to float if integral."""
    arr = np.asarray(arr)
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float64)
    return arr
