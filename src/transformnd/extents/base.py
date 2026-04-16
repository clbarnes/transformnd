from abc import ABC, abstractmethod


class Extents[ArrayT](ABC):
    """Base class for determining whether coordinates are "inside" a space."""

    ndim: set[int] | None = None

    @abstractmethod
    def contains(self, coords: ArrayT) -> ArrayT: ...
