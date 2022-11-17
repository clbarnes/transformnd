"""Bridging transforms between known spaces."""
from functools import lru_cache
from typing import Dict, Iterable, Iterator, Optional, Set, Tuple

import networkx as nx
import numpy as np

from .base import Transform, TransformSequence
from .util import SpaceRef, chain_or, dim_intersection, window


def split_sequence(seq: TransformSequence) -> Iterator[Transform]:
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
        if t.source_space is not None and t.target_space is not None:
            yield t
            continue

        this_seq.append(t)
        if t.target_space is not None:
            yield TransformSequence(this_seq)
            this_seq = []


class TransformGraph:
    def __init__(self):
        """Transform between any number of arbitrary spaces.

        Finds the shortest path for transforming one space
        into another, via some intermediate spaces.

        Populate with `my_transform_graph.add_transforms(my_transforms)`.

        """
        self.graph = nx.OrderedDiGraph()
        self.ndim: Optional[Set[int]] = None

    def add_transforms(self, transforms: Iterable[Transform]):
        """
        Parameters
        ----------
        transforms : Iterable[Transform]
            Transforms which must have a source and target space defined.
            TransformSequences are split out if their inner transforms'
            spaces are defined.

        Raises
        ------
        ValueError
            Undefined source and target spaces.
        """
        # TODO: weighting of split-out sequences could be problematic
        edges: Dict[Tuple[SpaceRef, SpaceRef], Transform] = dict()
        self.get_sequence.cache_clear()

        ndim = self.ndim

        for t in transforms:
            ndim = dim_intersection(ndim, t.ndim)
            if ndim is not None and len(ndim) == 0:
                raise ValueError("This TransformGraph supports no dimensionality")

            if isinstance(t, TransformSequence):
                ts = list(split_sequence(t))
            else:
                ts = [t]

            for t2 in ts:
                if chain_or(t.source_space, t.target_space, default=None) is None:
                    raise ValueError(
                        "All transforms in a graph "
                        "need explicit source and target spaces"
                    )
                edges[(t2.source_space, t2.target_space)] = t2

        self.ndim = ndim

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

        return count

    @lru_cache()
    def get_sequence(
        self, source_space: SpaceRef, target_space: SpaceRef
    ) -> TransformSequence:
        """Get the shortest TransformSequence for transforming between two spaces.

        Parameters
        ----------
        source_space : SpaceRef
        target_space : SpaceRef

        Returns
        -------
        TransformSequence
        """
        path = nx.shortest_path(self.graph, source_space, target_space)
        if len(path) <= 1:
            transforms = []
        else:
            transforms = [
                self.graph.edges[src, tgt]["transform"] for src, tgt in window(path, 2)
            ]
        return TransformSequence(
            transforms,
            spaces=(source_space, target_space),
        )

    def transform(
        self, source_space: SpaceRef, target_space: SpaceRef, coords: np.ndarray
    ) -> np.ndarray:
        """Transform coordinates from one space to another,
        possibly via intermediates.

        Parameters
        ----------
        source_space : SpaceRef
        target_space : SpaceRef
        coords : np.ndarray

        Returns
        -------
        np.ndarray
        """
        t = self.get_sequence(source_space, target_space)
        return t.apply(coords)

    def __iter__(self) -> Iterator[Transform]:
        """Iterate through the transforms present in the graph.

        Includes inferred reverse transforms.

        N.B. `networkx.Graph.__iter__` iterates through nodes,
        where this effectively iterates through edges.

        Yields
        -------
        Transform

        Examples
        --------
        Create a new transform graph using another

        >>> new_tgraph = TransformGraph([extra_transform, *old_tgraph])

        """
        for _, _, t in self.graph.edges.data("transform"):
            yield t
