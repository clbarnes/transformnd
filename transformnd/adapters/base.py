"""Simple adapter cases."""
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import partial
from typing import Callable, Generic, TypeVar

import numpy as np

from ..base import Transform

T = TypeVar("T")


class BaseAdapter(Generic[T], ABC):
    @abstractmethod
    def __call__(self, transform: Transform, obj: T) -> T:
        """Apply the given transformation to a non-array object.

        Parameters
        ----------
        transform : Transform
        obj : T

        Returns
        -------
        T
        """
        pass

    def partial(self, *args, **kwargs) -> Callable:
        """Create a partial function with frozen arguments.

        Useful for applying the same transform to many objects,
        or many transforms to the same object,
        or for adapters with additional arguments,
        using the same config repeatedly.

        Returns
        -------
        Callable
        """
        return partial(self, *args, **kwargs)


class NullAdapter(BaseAdapter[np.ndarray]):
    """Adapter which simply applies the transform."""

    def __call__(self, transform: Transform, obj: np.ndarray) -> np.ndarray:
        return transform(obj)


class FnAdapter(BaseAdapter[T]):
    def __init__(self, fn: Callable[[Transform, T], T]):
        """Adapter which simply wraps a function, for typing purposes.

        Parameters
        ----------
        fn : Callable[[Transform, T], T]
            Function which takes the object,
            and applies the transformation to it.
        """
        self.fn = fn

    def __call__(self, transform: Transform, obj: T) -> T:
        return self.fn(transform, obj)


class AttrAdapter(BaseAdapter[T]):
    def __init__(self, **kwargs: BaseAdapter) -> None:
        """Adapter which transforms an object by applying transforms to its member variables.

        Parameters
        ----------
        adapters : Dict[str, BaseAdapter]
            Keys are attribute names, values are adapters with which
            to apply the transform to those attributes.
        """
        self.adapters = kwargs

    def __call__(self, transform: Transform, obj: T, in_place=False) -> T:
        """Apply the given transformation to the object, via its attributes.

        Parameters
        ----------
        transform : Transform
        obj : T
        in_place : bool, optional
            Whether to mutate the given object in place,
            by default False (i.e. make a deep copy of it).

        Returns
        -------
        T
        """
        if not in_place:
            obj = deepcopy(obj)

        for k, v in self.adapters.items():
            member = getattr(obj, k)
            try:
                transformed = v(transform, member, in_place=True)  # type: ignore
            except TypeError as e:
                if "got an unexpected keyword argument 'in_place'" in str(e):
                    transformed = v(transform, member)
                else:
                    raise e
            setattr(obj, k, transformed)

        return obj


class SimpleAdapter(BaseAdapter, Generic[T], ABC):
    """
    Helper class for cases with simple conversion methods.
    """

    @abstractmethod
    def _to_array(self, obj: T) -> np.ndarray:
        """Convert the object into an array of coordinates."""
        pass

    @abstractmethod
    def _from_array(self, coords: np.ndarray) -> T:
        """Convert an array of coordinates into the correct type."""
        pass

    def __call__(self, transform: Transform, obj: T) -> T:
        coords = self._to_array(obj)
        transformed = transform(coords)
        return self._from_array(transformed)


class ReshapeAdapter(BaseAdapter[np.ndarray], ABC):
    """Adapter which reshapes a numpy.ndarray"""

    def __init__(self, dim_axis=-1) -> None:
        """Adapt numpy arrays which are not of the correct shape.

        Parameters
        ----------
        dim_axis : int, optional
            Which axis contains the coordinates' dimensions,
            by default -1 (last)
        """
        self.dim_axis: int = dim_axis

    def __call__(self, transform: Transform, arr: np.ndarray) -> np.ndarray:
        dim_axis = self.dim_axis
        if self.dim_axis < 0:
            dim_axis += arr.ndim

        moved = np.moveaxis(arr, dim_axis, -1)
        m_shape = moved.shape

        flattened = np.reshape(moved, (-1, m_shape[-1]))
        transformed = transform(flattened)
        return np.moveaxis(np.reshape(transformed, m_shape), -1, dim_axis)
