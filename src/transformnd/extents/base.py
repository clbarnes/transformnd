from abc import ABC, abstractmethod


class Extents[ArrayT](ABC):
    """Base class for determining whether coordinates are "inside" a space."""

    def __init__(self, ndim: int) -> None:
        self.ndim = ndim

    @abstractmethod
    def contains(self, coords: ArrayT) -> ArrayT: ...
