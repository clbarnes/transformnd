from functools import lru_cache
from typing import Dict, Iterable, Iterator, Tuple

import networkx as nx
import numpy as np

from .base import Transform, TransformSequence
from .util import SpaceRef, chain_or, window


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
    def __init__(self, transforms: Iterable[Transform]):
        """Transform between any number of arbitrary spaces.

        Finds the shortest path for transforming one space
        into another, via some intermediate spaces.

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
        self.graph = nx.OrderedDiGraph()

        edges: Dict[Tuple[SpaceRef, SpaceRef], Transform] = dict()
        for t in transforms:
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

        for (src, tgt), t in edges.items():
            self.graph.add_edge(src, tgt, transform=t)
            if (tgt, src) not in edges:
                try:
                    self.graph.add_edge(tgt, src, transform=-t)
                except NotImplementedError:
                    pass

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
            transforms, source_space=source_space, target_space=target_space
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
        return t(coords)

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
