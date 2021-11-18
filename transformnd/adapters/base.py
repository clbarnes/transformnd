from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Callable, Dict, Generic, TypeVar

import numpy as np

from ..base import Transform

T = TypeVar("T")


class BaseAdapter(Generic[T], ABC):
    @abstractmethod
    def __call__(self, transform: Transform, obj: T) -> T:
        pass


class NullAdapter(BaseAdapter[np.ndarray]):
    def __call__(self, transform: Transform, obj: np.ndarray) -> np.ndarray:
        return transform(obj)


class FnAdapter(BaseAdapter[T]):
    def __init__(self, fn: Callable[[Transform, T], T]) -> None:
        self.fn = fn

    def __call__(self, transform: Transform, obj: T) -> T:
        return self.fn(transform, obj)


class AttrAdapter(BaseAdapter):
    def __init__(self, adapters: Dict[str, BaseAdapter]) -> None:
        self.adapters = adapters

    def __call__(self, transform: Transform, obj: T, in_place=False) -> T:
        if not in_place:
            obj = deepcopy(obj)

        for k, v in self.adapters.items():
            member = getattr(obj, k)
            transformed = v(transform, member)
            setattr(obj, k, transformed)

        return obj


class Adapter(BaseAdapter, Generic[T], ABC):
    @abstractmethod
    def _to_array(self, obj: T) -> np.ndarray:
        pass

    @abstractmethod
    def _from_array(self, coords: np.ndarray) -> T:
        pass

    def __call__(self, transform: Transform, obj: T) -> T:
        coords = self._to_array(obj)
        transformed = transform(coords)
        return self._from_array(transformed)
