"""Bridging transforms between known spaces."""

from __future__ import annotations
from functools import lru_cache
from collections.abc import Callable, Iterator
import logging
from itertools import pairwise
from types import ModuleType
from typing import Any

import networkx as nx

from .transforms.bijection import Bijection
from .base import Transform, TransformSequence
from .util import SpaceRef, ArrayT, same_or_none
from .types import Spaces

logger = logging.getLogger(__name__)

TRANSFORM_KEY = "_transform"
WeightFn = Callable[[SpaceRef, SpaceRef, dict[str, Any]], int]


def split_sequence(seq: TransformSequence[ArrayT]) -> Iterator[Transform[ArrayT]]:
    """Split a TransformSequence into Transforms with spaces defined.

    If a component Transform has its spaces defined,
    it will be yielded as-is.
    A chain of Transforms without spaces defined are yielded as a TransformSequence.

    Parameters
    ----------
    seq
        The TransformSequence to split.

    Yields
    ------
    Transform[ArrayT]
        Individual transforms or subsequences with defined spaces.
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


def normalise_edge_weight_fn(w: str | WeightFn | None) -> WeightFn:
    if w is None:
        return lambda _s, _t, _d: 1
    elif isinstance(w, str):
        return lambda _s, _t, d: d.get(w, 1)
    else:
        return w


class TransformGraph[ArrayT]:
    """Transform between any number of arbitrary spaces/ coordinate systems.

    Finds the shortest path for transforming one space
    into another, via some intermediate spaces.

    Populate with `my_transform_graph.add_transforms(my_transforms)`.
    """

    def __init__(
        self,
    ):
        """Create an transform graph, optionally with some starting transforms.

        See the `TransformGraph.add_transforms` documentation for restrictions on the
        given transforms.
        """
        self.graph = nx.MultiDiGraph()
        self.space_ndims: dict[SpaceRef, int] = dict()

    def _update_spaces(
        self,
        transform: Transform[ArrayT],
        source: SpaceRef | None,
        target: SpaceRef | None,
    ) -> Spaces:
        """Check that the transform's spaces do not conflict with those given explicitly,
        that the source and target space is defined somewhere,
        and that the dimensionality of the spaces (inferred from the transforms)
        does not conflict with known spaces.
        """
        # check explicit spaces do not conflict with transform's spaces
        src = same_or_none(transform.spaces.source, source)
        tgt = same_or_none(transform.spaces.target, target)

        # if the node already exists, make sure the dimensionality does not conflict
        self.space_ndims[src] = same_or_none(
            self.space_ndims.get(src), transform.ndims.source
        )
        self.space_ndims[tgt] = same_or_none(
            self.space_ndims.get(tgt), transform.ndims.target
        )
        return Spaces(src, tgt)

    def _add_transform(
        self,
        transform: Transform[ArrayT],
        source: SpaceRef | None,
        target: SpaceRef | None,
        edge_data: dict[str, Any] | None,
    ) -> list[tuple[SpaceRef, SpaceRef]]:
        """Clearing the get_sequence cache and splitting sequences and bijections should be handled outside this method."""
        out = []

        src, tgt = self._update_spaces(transform, source, target)

        if edge_data is None:
            edge_data = dict()

        if TRANSFORM_KEY in edge_data:
            raise ValueError(f"Must not use the key '{TRANSFORM_KEY}' in edge_data")

        d = {TRANSFORM_KEY: transform, **edge_data}
        self.graph.add_edge(src, tgt, **d)
        out.append((src, tgt))
        return out

    def add_transform(
        self,
        transform: Transform[ArrayT],
        source: SpaceRef | None = None,
        target: SpaceRef | None = None,
        *,
        edge_data: dict[str, Any] | None = None,
    ) -> list[tuple[SpaceRef, SpaceRef]]:
        """Add a transform to the graph.

        If the given transform is a `Bijection`,
        only the forward component will be added as an independent edges.

        This method will NOT overwrite intermediate edges.

        N.B. Previously this method implicitly added inverse edges where possible.
        Now these edges must be added explicitly by calling `add_transform(~transform)`.
        Additionally, previously `TransformSequence`s would be split out into multiple edges
        if any intermediate spaces were defined;
        now these edges must be added explicitly with the `TransformSequence.split` method.

        Parameters
        ----------
        transform
            Transform to add to the graph as an edge.
        source
            May be omitted if `transform` has its source space defined.
        target
            May be omitted if `transform` has its target space defined.
        edge_data
            Dict of string keys to arbitrary values to associate with an edge.
            Used during path-finding.
            Must not have the `"_transform"` key.

        Returns
        -------
        list[tuple[SpaceRef, SpaceRef]]
            List of `(src, tgt)` edges added to the graph.
        """
        out: list[tuple[SpaceRef, SpaceRef]] = []
        if isinstance(transform, Bijection):
            out.extend(
                self.add_transform(
                    transform.forward,
                    source,
                    target,
                    edge_data=edge_data,
                )
            )

        else:
            out.extend(self._add_transform(transform, source, target, edge_data))

        if out:
            self.get_sequence.cache_clear()

        return out

    @lru_cache()
    def get_sequence(
        self,
        source_space: SpaceRef,
        target_space: SpaceRef,
        full: bool = False,
        *,
        weight: None | str | WeightFn = None,
    ) -> TransformSequence[ArrayT]:
        """Get the shortest TransformSequence for transforming between two spaces.

        Parameters
        ----------
        source_space
            The source coordinate space.
        target_space
            The target coordinate space.
        full
            By default, simplifies consecutive affines and drops bijections' inverse form.
            If `full` is True, keeps each transformation as-is.
        weight
            str key in the `edge_data` dict given when an edge was added,
            or a function to determine a weight from the args `src_space, tgt_space, edge_data`,
            or None (all weights are 1).

        Returns
        -------
        TransformSequence[ArrayT]
            The shortest transform sequence between the spaces.
        """
        path = nx.shortest_path(self.graph, source_space, target_space, weight)  # type:ignore
        transforms = []
        wfn = normalise_edge_weight_fn(weight)

        for src, tgt in pairwise(path):
            edges = self.graph[src][tgt]
            transforms.append(
                min(edges.values(), key=lambda d: wfn(src, tgt, d))[TRANSFORM_KEY]
            )

        seq = TransformSequence(
            transforms,
            spaces=Spaces(source_space, target_space),
        )
        if not full:
            seq = seq.simplify(drop_inverse=True)
        return seq

    def transform(
        self,
        source_space: SpaceRef,
        target_space: SpaceRef,
        coords: ArrayT,
        *,
        weight: None | str | WeightFn = None,
    ) -> ArrayT:
        """Transform coordinates from one space to another,
        possibly via intermediates.

        Parameters
        ----------
        source_space
            The source coordinate space.
        target_space
            The target coordinate space.
        coords
            The coordinates to transform.
        weight
            str key in the `edge_data` dict given when an edge was added,
            or a function to determine a weight from the args `src_space, tgt_space, edge_data`,
            or None (all weights are 1).


        Returns
        -------
        ArrayT
            The transformed coordinates.
        """
        t = self.get_sequence(source_space, target_space, weight=weight)
        return t.apply(coords)

    def __iter__(self) -> Iterator[Transform[ArrayT]]:
        """Iterate through the transforms present in the graph.

        Includes inferred reverse transforms.

        N.B. the `__iter__` method of some popular graph libraries like networkx
        iterate through nodes, where this effectively iterates through edges.

        Yields
        ------
        Transform[ArrayT]
            The next transform in the graph.

        Examples
        --------
        Create a new transform graph using another

        >>> new_tgraph = TransformGraph([extra_transform, *old_tgraph])

        """
        for _, _, t in self.graph.edges.data(TRANSFORM_KEY):
            yield t

    def to_device(
        self, xp: ModuleType, device: str | None = None
    ) -> TransformGraph[ArrayT]:
        result: TransformGraph[ArrayT] = TransformGraph()
        for src, tgt, t in self.graph.edges.data(TRANSFORM_KEY):
            result.graph.add_edge(src, tgt, transform=t.to_device(xp, device))
        return result
