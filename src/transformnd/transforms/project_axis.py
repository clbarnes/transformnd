from __future__ import annotations
from typing import Self

import numpy as np
from array_api_compat import array_namespace
from transformnd.transforms import Affine
from transformnd.types import NDims
from ..base import Transform
from ..types import ArrayT


class ProjectAxis(Transform):
    """Transform for adding and removing axes.

    WARNING: inverting this transformation may be lossy.
    """

    def __init__(
        self,
        dropped: set[int] | None = None,
        created: set[int] | None = None,
        source_ndim: int | None = None,
        target_ndim: int | None = None,
    ):
        """Create a transform for adding and dropping axes.

        At least one of source_ndim and target_ndim must be given.

        Parameters
        ----------
        dropped
            Set of INPUT dimension indices to drop, if any.
        created
            Set of OUTPUT dimension indices which are new, if any.
        source_ndim
            If omitted, can be inferred from `target_ndim`.
        target_ndim
            If omitted, can be inferred from `source_ndim`.

        Raises
        ------
        ValueError
            Operations are inconsistent with given dimensionality,
            or insufficient dimensionality information was given.
        """
        self.dropped = dropped or set()
        self.created = created or set()

        if source_ndim is not None:
            nd = source_ndim - len(self.dropped) + len(self.created)
            if target_ndim is None:
                target_ndim = nd
            elif target_ndim != nd:
                raise ValueError("Operations do not match expected target ndim")

        elif target_ndim is not None:
            nd = target_ndim - len(self.created) + len(self.dropped)
            if source_ndim is None:
                source_ndim = nd
            elif source_ndim != nd:
                raise ValueError("Operations do not match expected source ndim")

        else:
            raise ValueError("At least one of source_ndim or target_ndim must be given")

        idxs: list[int | None] = list(range(source_ndim))
        for drop in sorted(self.dropped, reverse=True):
            idxs.pop(drop)
        for create in sorted(self.created):
            idxs.insert(create, None)
        self._idxs = idxs

        super().__init__(NDims(source_ndim, target_ndim))

    def apply(self, coords: ArrayT) -> ArrayT:
        coords = self._validate_coords(coords)
        if self.created:
            xp = array_namespace(coords)
            out = xp.zeros_like(coords, shape=(xp.shape(coords)[0], self.ndims.target))
            for idx, orig_idx in enumerate(self._idxs):
                if orig_idx is not None:
                    out[:, idx] = coords[:, orig_idx]  # type:ignore

        else:
            out = coords[:, self._idxs]  # type:ignore
        return out

    def is_identity(self) -> bool:
        return not self.created and not self.dropped

    def to_affine(self) -> Affine | None:
        m = np.eye(self.ndims.source)
        out_m = self.apply(m)
        return Affine.from_linear_map(out_m.T)

    def invert(self) -> Self | None:
        return type(self)(
            self.created,
            self.dropped,
            source_ndim=self.ndims.target,
            target_ndim=self.ndims.source,
        )
