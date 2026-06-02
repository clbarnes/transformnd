"""Bridging transforms between known spaces."""

from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from collections.abc import Iterable, Iterator

import networkx as nx

from .base import Transform, TransformSequence
from .util import SpaceRef, chain_or, ArrayT
from .types import Spaces
from itertools import pairwise


def split_sequence(seq: TransformSequence[ArrayT]) -> Iterator[Transform[ArrayT]]:
    """Split a TransformSequence into Transforms with spaces defined.

    If a component Transform has its spaces defined,
    it will be yielded as-is.
    A chain of Transforms without spaces defined are yielded as a TransformSequence.

    Parameters
    ----------
    seq : TransformSequence

    Yields
    -------
    Transform
    """
    this_seq = []
    for t in seq.transforms:
        if t.spaces.source is not None and t.spaces.target is not None:
            yield t
            continue

        this_seq.append(t)
        if t.spaces.target is not None:
            yield TransformSequence(this_seq)
            this_seq = []


@dataclass(frozen=True, eq=True)
class SimplifyConfig:
    ndim: int | None = None
    """Force specific dimensionality, allowing conversion to affines."""

    drop_inverse: bool = False
    """Drop explicit inverses in bijection transformations."""


class NDimRegistries:
    def __init__(self, perm: dict[SpaceRef, int]) -> None:
        self.perm = perm
        self.temp: dict[SpaceRef, int] = dict()

    def _check_inner(
        self, space: SpaceRef, ndim: int, reg: dict[SpaceRef, int], add: bool = False
    ):
        val = reg.get(space)
        if val is None:
            if add:
                reg[space] = ndim
            return None
        if val != ndim:
            raise ValueError(
                f"New transform implies space {space} is {ndim}D, but it is already registered as {val}D"
            )

    def merge(self):
        self.perm.update(self.temp)

    def check(self, space: SpaceRef, ndim: int):
        self._check_inner(space, ndim, self.perm)
        self._check_inner(space, ndim, self.temp, True)


class TransformGraph[ArrayT]:
    """Transform between any number of arbitrary spaces/ coordinate systems.

    Finds the shortest path for transforming one space
    into another, via some intermediate spaces.

    Populate with `my_transform_graph.add_transforms(my_transforms)`.
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.space_ndims: dict[SpaceRef, int] = dict()

    def add_transforms(self, transforms: Iterable[Transform[ArrayT]]) -> int:
        """Bulk-add transformations to the graph.

        Every given transform must have a source and target space defined;
        these spaces are the nodes of the graph.

        If any of the given transforms are `TransformSequence`s,
        any subsequences of transformations with source and target spaces
        (explicitly defined or implicit based on their neighbours')
        will be split into separate graph edges.

        Note that a single `TransformSequence` is itself an `Iterable[Transform]`
        and so could be used as the `transforms` argument.
        However, a `TransformSequence` does not require that all of its members
        have explicit source and target spaces,
        where the `transforms` argument here does,
        so not all `TransformSequence`s can be used directly as the argument
        (wrap them in a list instead).

        Parameters
        ----------
        transforms : Iterable[Transform[ArrayT]]
            Transforms which must have a source and target space defined.
            TransformSequences are split out if their inner transforms'
            spaces are defined.

        Raises
        ------
        ValueError
            Undefined source and target spaces.
        """
        # TODO: weighting of split-out sequences could be problematic
        edges: dict[tuple[SpaceRef, SpaceRef], Transform[ArrayT]] = dict()
        self.get_sequence.cache_clear()

        registry = NDimRegistries(self.space_ndims)

        for t in transforms:
            if isinstance(t, TransformSequence):
                ts = list(split_sequence(t))
            else:
                ts = [t]

            for t2 in ts:
                if chain_or(t2.spaces.source, t2.spaces.target, default=None) is None:
                    raise ValueError(
                        "All transforms in a graph "
                        "need explicit source and target spaces"
                    )
                registry.check(t2.spaces.source, t2.ndims.source)
                registry.check(t2.spaces.target, t2.ndims.target)
                edges[(t2.spaces.source, t2.spaces.target)] = t2

        count = 0

        for (src, tgt), t in edges.items():
            self.graph.add_edge(src, tgt, transform=t)
            count += 1
            if (tgt, src) not in edges:
                try:
                    self.graph.add_edge(tgt, src, transform=~t)
                    count += 1
                except NotImplementedError:
                    pass

        registry.merge()

        return count

    @lru_cache()
    def get_sequence(
        self,
        source_space: SpaceRef,
        target_space: SpaceRef,
        full: bool = False,
    ) -> TransformSequence[ArrayT]:
        """Get the shortest TransformSequence for transforming between two spaces.

        Parameters
        ----------
        source_space : SpaceRef
        target_space : SpaceRef
        full : bool
            By default, simplifies consecutive affines and drops bijections' inverse form.
            If `full` is True, keeps each transformation as-is.

        Returns
        -------
        TransformSequence[ArrayT]
        """
        path = nx.shortest_path(self.graph, source_space, target_space)
        if len(path) <= 1:
            transforms = []
        else:
            transforms = [
                self.graph.edges[src, tgt]["transform"] for src, tgt in pairwise(path)
            ]
        seq = TransformSequence(
            transforms,
            spaces=Spaces(source_space, target_space),
        )
        if not full:
            seq = seq.simplify(drop_inverse=False)
        return seq

    def transform(
        self,
        source_space: SpaceRef,
        target_space: SpaceRef,
        coords: ArrayT,
    ) -> ArrayT:
        """Transform coordinates from one space to another,
        possibly via intermediates.

        Parameters
        ----------
        source_space : SpaceRef
        target_space : SpaceRef
        coords : ArrayT

        Returns
        -------
        ArrayT
        """
        t = self.get_sequence(source_space, target_space)
        return t.apply(coords)

    def __iter__(self) -> Iterator[Transform[ArrayT]]:
        """Iterate through the transforms present in the graph.

        Includes inferred reverse transforms.

        N.B. the `__iter__` method of some popular graph libraries like networkx iterate through nodes,
        where this effectively iterates through edges.

        Yields
        -------
        Transform[ArrayT]

        Examples
        --------
        Create a new transform graph using another

        >>> new_tgraph = TransformGraph([extra_transform, *old_tgraph])

        """
        for _, _, t in self.graph.edges.data("transform"):
            yield t

    def to_device(self, xp, device=None) -> TransformGraph[ArrayT]:
        result: TransformGraph[ArrayT] = TransformGraph()
        for src, tgt, t in self.graph.edges.data("transform"):
            result.graph.add_edge(src, tgt, transform=t.to_device(xp, device))
        return result
