from collections import deque
from typing import (
    Any,
    Callable,
    Deque,
    Hashable,
    Iterable,
    Iterator,
    Optional,
    Set,
    Tuple,
)

import numpy as np

UNSPECIFIED_SPACE_NAME = "???"

TransformSignature = Callable[[np.ndarray], np.ndarray]
"""Type annotation of a function which can be used as a transform."""

SpaceRef = Hashable
"""Type annotation of things which can be used to refer to spaces"""


def none_eq(a: Optional[Any], b: Optional[Any]) -> bool:
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


def chain_or(*args: Optional[Any], default=NO_DEFAULT):
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


def same_or_none(*args, default=NO_DEFAULT):
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


def window(iterable: Iterable, length: int) -> Iterator[Tuple[Any, ...]]:
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
    q: Deque[Any] = deque(maxlen=length)
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


def flatten(
    arr,
    dim_axis=-1,
    transpose=False,
) -> Tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """Convert array into 2D, and provide a routine to reclaim original shape.

    Parameters
    ----------
    arr : np.ndarray
        Array to be flattened
    dim_axis : bool, optional
        Which axis has the different dimensions

    Returns
    -------
    Tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]
        The flattened array, and a function
        for converting an identically-shaped flat array
        into one the same shape as the original.

    Example
    -------
    >>> my_coords = np.random.random((3, 30, 20))
    >>> flat, unflatten = flatten(my_coords)
    >>> flat.shape == (3, 30*20)
    >>> recovered = unflatten(flat)
    >>> np.allclose(recovered, my_coords)
    """
    # todo: reduce copies if possible
    if dim_axis < 0:
        dim_axis = arr.ndim + dim_axis
    moved = np.moveaxis(arr, dim_axis, -1)
    m_shape = moved.shape

    flattened = np.reshape(moved, (-1, m_shape[-1]))
    if transpose:
        flattened = flattened.T

    def unflatten(flat):
        if transpose:
            flat = flat.T
        return np.moveaxis(np.reshape(flat, m_shape), -1, dim_axis)

    return flattened, unflatten


def check_ndim(given_ndim: int, supported_ndim: Optional[Set[int]]):
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


def format_dims(supported):
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
        return "null-D"
    return "/".join(f"{d}D" for d in sorted(supported))


def space_str(space: Optional[SpaceRef]):
    if space is None:
        return UNSPECIFIED_SPACE_NAME
    else:
        return str(space)


def is_square(arr: np.ndarray) -> bool:
    return arr.ndim == 2 and arr.shape[0] == arr.shape[1]
