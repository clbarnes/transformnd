from collections.abc import Iterable, Sequence
from typing import Self, Generic
from .base import Transform
from .types import ArrayT, SrcTgt
from .types import SpaceRef


class Spaced(Generic[ArrayT, SpaceRef]):
    def __init__(
        self, transform: Transform[ArrayT], source: SpaceRef, target: SpaceRef
    ) -> None:
        self.transform = transform
        self.spaces: SrcTgt[SpaceRef] = SrcTgt(source=source, target=target)

    def __or__(self, rhs: Self) -> Self:
        if not isinstance(rhs, Spaced):
            return NotImplemented

        if not self.precedes(rhs):
            raise ValueError("target and source space mismatch")

        t = self.transform | rhs.transform
        return type(self)(t, self.spaces.source, rhs.spaces.target)

    def __ror__(self: Self, lhs: Self) -> Self:
        if not isinstance(lhs, Spaced):
            return NotImplemented

        return lhs | self

    def precedes(self, antecedent: Self) -> bool:
        return self.spaces.target == antecedent.spaces.source

    def follows(self, precedent: Self) -> bool:
        return precedent.spaces.target == self.spaces.source

    def invert(self) -> Self | None:
        t = self.transform.invert()
        if t is None:
            return None
        return type(self)(t, self.spaces.source, self.spaces.target)

    def __invert__(self):
        out = self.invert()
        if out is None:
            return NotImplemented
        return out


def valid_path(transforms: Sequence[Spaced]) -> bool:
    """Check whether a sequence of spaced transforms is valid."""
    try:
        for _ in iter_valid_path(transforms):
            pass
        return True
    except ValueError:
        return False


def iter_valid_path(transforms: Iterable[Spaced]) -> Iterable[Spaced]:
    """Yield transforms, raising an exception if the spaces do not match."""
    prev = None
    for t in transforms:
        if prev is not None and not t.follows(prev):
            raise ValueError(f"Mismatched transform spaces: {prev} -> {t}")
        prev = t
        yield prev
