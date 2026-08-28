"""Bridging transforms between known spaces."""

from __future__ import annotations
from copy import copy, deepcopy
from functools import lru_cache
from collections.abc import Callable, Iterable, Iterator, Mapping
import logging
from itertools import pairwise
from types import ModuleType
from typing import Any, Generic, Hashable, Self
from typing_extensions import TypeVar
from .spaced import Spaced

import networkx as nx

from transformnd.transforms.simple import Identity

from .transforms.bijection import Bijection
from .base import Transform, TransformSequence
from .util import ArrayT
from .types import SpaceRef

logger = logging.getLogger(__name__)

TRANSFORM_KEY = "_transform"
NDIM_KEY = "ndim"
WeightFn = Callable[[SpaceRef, SpaceRef, dict[str, Any]], int]

SpaceRef2 = TypeVar("SpaceRef2", bound=Hashable, default=Hashable)


def normalise_edge_weight_fn(w: str | WeightFn | None) -> WeightFn:
    if w is None:
        return lambda _s, _t, _d: 1
    elif isinstance(w, str):
        return lambda _s, _t, d: d.get(w, 1)
    else:
        return w


class TransformGraph(Generic[ArrayT, SpaceRef]):
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

    def ndim(self, space: SpaceRef) -> int | None:
        """Get the dimensionality of the given space, or None if missing."""
        d = self.graph.nodes.get(space)
        if d is None:
            return None
        return d["ndim"]

    def copy(self, deep=False) -> Self:
        """Take an (optionally deep) copy of the graph."""
        if deep:
            return deepcopy(self)
        else:
            return copy(self)

    def relabel_spaces(
        self, mapping: Mapping[SpaceRef, SpaceRef2] | Callable[[SpaceRef], SpaceRef2]
    ) -> TransformGraph[ArrayT, SpaceRef2]:
        """Relabel space references.

        This is useful when e.g. merging graphs and wanting to make sure their spaces do not clash,
        by appending different suffixes to spaces from each graph.

        Edge data are shallow-copied.

        Parameters
        ----------
        mapping
            Mapping from the old space references to the new.
            May be a mapping (e.g. a dict) or a callable (e.g. a function).
            Outputs of the mapping may re-use labels from the inputs,
            but no validation

        Returns
        -------
        TransformGraph[ArrayT, SpaceRef2]
            May use a different type for space references.
        """
        g = TransformGraph[ArrayT, SpaceRef2]()
        if isinstance(mapping, Mapping):

            def f(k: SpaceRef, /) -> SpaceRef2:
                return mapping[k]
        else:
            f = mapping

        for s, d in self:
            mapped = Spaced(s.transform, f(s.spaces.source), f(s.spaces.target))
            g.add_transform(mapped, edge_data=d)
        return g

    def _add_space(self, space: SpaceRef, ndim: int):
        curr_ndim = self.ndim(space)
        if ndim is None:
            self.graph.add_node(space, ndim=ndim)
        elif curr_ndim != ndim:
            raise ValueError(f"Space {space} is {curr_ndim}D, got {ndim}D")

    def _add_transform(
        self,
        spaced: Spaced[ArrayT, SpaceRef],
        edge_data: dict[str, Any] | None,
    ) -> list[tuple[SpaceRef, SpaceRef]]:
        """Clearing the get_sequence cache and splitting sequences and bijections should be handled outside this method."""
        out = []
        t, source, target = (
            spaced.transform,
            spaced.spaces.source,
            spaced.spaces.target,
        )
        self._add_space(source, t.ndims.source)
        self._add_space(target, t.ndims.target)

        if edge_data is None:
            edge_data = dict()

        if TRANSFORM_KEY in edge_data:
            raise ValueError(f"Must not use the key '{TRANSFORM_KEY}' in edge_data")

        d = {TRANSFORM_KEY: t, **edge_data}
        self.graph.add_edge(source, target, **d)
        out.append((source, target))
        return out

    def add_transform(
        self,
        spaced: Spaced[ArrayT, SpaceRef],
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
        spaced
            Spaced transform to add to the graph as an edge.
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
        inner, src, tgt = (
            spaced.transform,
            spaced.spaces.source,
            spaced.spaces.target,
        )
        if isinstance(inner, Bijection):
            out.extend(
                self.add_transform(
                    Spaced(inner.forward, src, tgt),
                    edge_data=edge_data,
                )
            )

        else:
            out.extend(self._add_transform(spaced, edge_data))

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
        if len(path) == 1:
            # source == target
            ndim = self.ndim(source_space)
            assert ndim is not None
            transforms.append(Identity[ArrayT](ndim))
        else:
            wfn = normalise_edge_weight_fn(weight)

            for src, tgt in pairwise(path):
                edges = self.graph[src][tgt]
                transforms.append(
                    min(edges.values(), key=lambda d: wfn(src, tgt, d))[TRANSFORM_KEY]
                )

        seq = TransformSequence(transforms)
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

    def __iter__(self) -> Iterator[tuple[Spaced[ArrayT, SpaceRef], dict]]:
        """Iterate through the transforms present in the graph,
        and a shallow copy of the edge data.

        N.B. the `__iter__` method of some popular graph libraries like networkx
        iterate through nodes, where this effectively iterates through edges.

        Yields
        ------
        tuple[Spaced[ArrayT, SpaceRef], dict]
            The spaced transform.
        """
        for src, tgt, data in self.graph.edges.data(True):
            d2 = data.copy()
            t = d2.pop(TRANSFORM_KEY)
            yield (Spaced(t, src, tgt), d2)

    def space_ndims(self) -> Iterable[tuple[SpaceRef, int]]:
        """Get the keys and dimensionality of all spaces."""
        return self.graph.nodes.data("ndim")

    def to_device[ArrayT2](
        self, xp: ModuleType, device: str | None = None
    ) -> TransformGraph[ArrayT2, SpaceRef]:
        result: TransformGraph[ArrayT2, SpaceRef] = TransformGraph()
        for src, tgt, d in self.graph.edges.data():
            t: Transform[ArrayT] = d.pop(TRANSFORM_KEY)
            t2: Transform[ArrayT2] = t.to_device(xp, device)  # type: ignore
            result.add_transform(Spaced(t2, src, tgt), edge_data=d)
        return result
