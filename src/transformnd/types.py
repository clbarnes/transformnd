from collections.abc import Sequence
from typing import Callable, Hashable, NamedTuple, Self
from typing_extensions import TypeVar
import numpy as np

from .constants import UNSPECIFIED_SPACE_NAME

ArrayT = TypeVar("ArrayT", default=np.ndarray)

type TransformSignature[ArrayT] = Callable[[ArrayT], ArrayT]
"""Type annotation of a function which can be used as a transform."""

type SpaceRef = Hashable
"""Type annotation of identifiers which can be used to refer to spaces"""


class SrcTgt[T](NamedTuple):
    """2-tuple where the two members represent some feature of a source and target space."""

    source: T
    target: T

    def invert(self) -> Self:
        return type(self)(self.target, self.source)

    @classmethod
    def from_seq(cls, sequence: Sequence[T]) -> Self:
        if len(sequence) != 2:
            raise ValueError("Expected 2-length sequence")
        return cls(sequence[0], sequence[1])

    def __str__(self) -> str:
        return f"{self.source}->{self.target}"


class Spaces(SrcTgt[SpaceRef | None]):
    """Source-target tuple for space identifiers."""

    def __str__(self) -> str:
        s = UNSPECIFIED_SPACE_NAME if self.source is None else self.source
        t = UNSPECIFIED_SPACE_NAME if self.target is None else self.source
        return f"{s}->{t}"


class NDims(SrcTgt[int]):
    """Source-target tuple for numbers of dimensions."""
