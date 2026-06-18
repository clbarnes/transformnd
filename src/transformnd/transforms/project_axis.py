from __future__ import annotations
from abc import ABC, abstractmethod
from copy import copy
from typing import Self, Sequence

import numpy as np
from array_api_compat import array_namespace
from transformnd.transforms import Affine
from transformnd.types import NDims, Spaces
from dataclasses import dataclass
from ..base import Transform
from ..types import ArrayT


@dataclass(frozen=True, eq=True)
class BaseOperation(ABC):
    idx: int
    """Which axis to apply the operation to."""

    def __post_init__(self):
        if self.idx < 0:
            raise ValueError("insert/remove idx must be positive")

    @abstractmethod
    def check(self, ndim: int) -> int: ...

    @abstractmethod
    def invert(self) -> Self: ...


@dataclass(frozen=True, eq=True)
class Insert(BaseOperation):
    """Component of the `ProjectAxis` transform which inserts a new axis."""

    def check(self, ndim: int) -> int:
        if self.idx > ndim or self.idx <= -ndim:
            raise ValueError(
                f"Index {self.idx} is out of range for dimensionality {ndim}"
            )
        return ndim + 1

    def invert(self) -> Remove:
        return Remove(self.idx)


@dataclass(frozen=True, eq=True)
class Remove(BaseOperation):
    """Component of the `ProjectAxis` transform which removes an existing axis."""

    def check(self, ndim: int) -> int:
        if self.idx >= ndim or self.idx <= -ndim:
            raise ValueError(
                f"Index {self.idx} is out of range for dimensionality {ndim}"
            )
        return ndim - 1

    def invert(self) -> Insert:
        if self.idx == -1:
            raise ValueError("Removal of the -1th axis is not invertible")
        return Insert(self.idx)


Operation = Insert | Remove
"""Insert or remove an axis."""


class ProjectAxis(Transform):
    """Transform for adding and removing axes.

    WARNING: inverting this transformation may be lossy.
    """

    def __init__(
        self,
        operations: Sequence[Operation],
        source_ndim: int | None = None,
        target_ndim: int | None = None,
        *,
        spaces: Spaces = Spaces(None, None),
    ):
        """Create a transform for adding and dropping axes.

        At least one of source_ndim and target_ndim must be given.

        Parameters
        ----------
        operations
            Sequence of operations to apply.
        source_ndim
            If omitted, can be inferred from `target_ndim`.
        target_ndim
            If omitted, can be inferred from `source_ndim`.
        spaces
            Identifiers for source and target spaces, by default Spaces(None, None)

        Raises
        ------
        ValueError
            Operations are inconsistent with given dimensionality,
            or insufficient dimensionality information was given.
        """
        self.operations = []
        self._has_inserts = False

        if source_ndim is not None:
            nd = source_ndim
            for op in operations:
                nd = op.check(nd)
            if target_ndim is None:
                target_ndim = nd
            elif target_ndim != nd:
                raise ValueError("Operations do not match expected target ndim")

        elif target_ndim is not None:
            nd = target_ndim
            for op in reversed(operations):
                nd = op.invert().check(nd)
            if source_ndim is None:
                source_ndim = nd
            elif source_ndim != nd:
                raise ValueError("Operations do not match expected source ndim")

        else:
            raise ValueError("At least one of source_ndim or target_ndim must be given")

        idxs: list[int | None] = list(range(source_ndim))
        for op in operations:
            if isinstance(op, Insert):
                self._has_inserts = True
                idxs.insert(op.idx, None)
            elif isinstance(op, Remove):
                idxs.pop(op.idx)
            self.operations.append(op)
        self._idxs = idxs

        super().__init__(NDims(source_ndim, target_ndim), spaces=spaces)

    def apply(self, coords: ArrayT) -> ArrayT:
        coords = self._validate_coords(coords)
        if self._has_inserts:
            xp = array_namespace(coords)
            out = xp.zeros_like(coords, shape=(coords.shape[0], self.ndims.target))
            for idx, orig_idx in enumerate(self._idxs):
                if orig_idx is not None:
                    out[:, idx] = coords[:, orig_idx]

        else:
            out = coords[:, self._idxs]
        return out

    def is_identity(self) -> bool:
        orig: list[int | None] = list(range(self.ndims.source))
        dims = copy(orig)
        for op in self.operations:
            if isinstance(op, Insert):
                dims.insert(op.idx, None)
            elif isinstance(op, Remove):
                dims.pop(op.idx)

        return dims == orig

    def to_affine(self) -> Affine | None:
        m = np.eye(self.ndims.source)
        out_m = self.apply(m)
        return Affine.from_linear_map(out_m.T)

    def invert(self) -> Self | None:
        return type(self)(
            [op.invert() for op in reversed(self.operations)],
            source_ndim=self.ndims.target,
            target_ndim=self.ndims.source,
            spaces=self.spaces.invert(),
        )
