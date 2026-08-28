"""Utilities used elsewhere in the package."""

from collections.abc import Iterable
import os
from types import ModuleType
import warnings
import logging

from array_api_compat import (
    array_namespace,
    is_dask_namespace,
    is_jax_namespace,
)
import numpy as np

from .types import ArrayT

logger = logging.getLogger(__name__)


def none_eq[T](a: T | None, b: T | None) -> bool:
    """Check whether either is None or both are equal.

    Parameters
    ----------
    a :
        First argument to check
    b :
        Second argument to check

    Returns
    -------
    bool
    """
    return a == b or a is None or b is None


class _NoDefault:
    pass


NO_DEFAULT = _NoDefault()


def chain_or[T](*args: T | None, default: _NoDefault | T = NO_DEFAULT) -> T:
    """Return the first of *args which is not None.

    Can either error or return a default if there are no non-None args.

    Parameters
    ----------
    *args
        Optional arguments to check.
    default
        By default, raises a ValueError if *args are exhausted.
        If given, returns the given value instead.

    Returns
    -------
    T
        One of the given args, or the default.

    Raises
    ------
    ValueError
        If `default` is not given and there are no non-None args.
    """
    for arg in args:
        if arg is not None:
            return arg
    if isinstance(default, _NoDefault):
        raise ValueError("No non-None arguments")
    return default


def same_or_none[T](*args: T | None, default: T | _NoDefault = NO_DEFAULT) -> T:
    """Check args are the same or None.

    If so, return the non-None value.
    Otherwise, raise a ValueError.

    Parameters
    ----------
    *args
        Arguments to check.
    default
        If given, return this instead of an error
        if all *args are None.

    Returns
    -------
    T
        The non-None arg value.

    Raises
    ------
    ValueError
        Arguments differ (other than Nones), or no non-None arguments are given (without a default).
    """
    prev = None

    for arg in args:
        if arg is None:
            continue
        if prev is not None and prev != arg:
            raise ValueError("Arguments are not None or the same")
        prev = arg

    if prev is None:
        if isinstance(default, _NoDefault):
            raise ValueError("No non-None arguments found")
        return default

    return prev


def format_dims(supported: set[int] | None) -> str:
    """Format supported dimensions for e.g. error messages.

    Parameters
    ----------
    supported
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


def as_floats(arr, *, namespace: ModuleType | None = None, device: str | None = None):
    """Get array-like as an array of floats.

    Convert to a particular array namespace if given;
    default to keeping the same namespace one exists,
    or numpy otherwise.

    Cast to a float if integral.
    """
    try:
        orig_namespace = array_namespace(arr)
    except TypeError:
        orig_namespace = None

    if namespace is None:
        namespace = orig_namespace or np

    kwargs = dict()
    if is_dask_namespace(namespace):
        if isinstance(device, str) and device != "cpu":
            logger.warning(f"Ignoring unsupported dask device: {device}")
    elif device is not None:
        kwargs["device"] = device

    arr = namespace.asarray(arr, **kwargs)  # type: ignore

    # dask does not have isdtype
    isdtype = getattr(namespace, "isdtype", np.isdtype)

    if not isdtype(arr.dtype, "real floating"):  # type:ignore
        if is_jax_namespace(namespace):
            dt = "float32"
        else:
            dt = "float64"
        # N.B. a dask array wrapping over jax arrays will warn here
        arr = arr.astype(dt, **kwargs)  # type:ignore

    return arr  # type:ignore


def set_scipy_array_api() -> bool:
    curr = os.environ.get("SCIPY_ARRAY_API")
    match curr:
        case None:
            os.environ["SCIPY_ARRAY_API"] = "1"
            return True
        case "1":
            return True
        case _:
            warnings.warn(
                "SCIPY_ARRAY_API environment set but not '1'; certain transforms may not work with certain array types"
            )
            return False


def join_strs(elems: Iterable, sep: str = ",", surround=("", "")) -> str:
    return surround[0] + sep.join(str(e) for e in elems) + surround[1]
