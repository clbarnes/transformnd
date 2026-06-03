"""Bridging transforms between known spaces."""

from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from collections.abc import Iterable, Iterator
import warnings
import logging
from itertools import chain, pairwise

import networkx as nx

from .transforms.bijection import Bijection
from .base import Transform, TransformSequence
from .util import SpaceRef, ArrayT, same_or_none
from .types import Spaces

logger = logging.getLogger(__name__)


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

    def __init__(
        self, transforms: Iterable[Transform[ArrayT]] | None = None, invert=True
    ):
        """Create an transform graph, optionally with some starting transforms.

        See the `TransformGraph.add_transforms` documentation for restrictions on the
        given transforms.
        """
        self.graph = nx.DiGraph()
        self.space_ndims: dict[SpaceRef, int] = dict()
        if transforms is not None:
            self.add_transforms(transforms, invert)

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
        invert: bool,
    ) -> list[tuple[SpaceRef, SpaceRef]]:
        """Clearing the get_sequence cache and splitting sequences should be handled outside this method."""
        count = []
        if isinstance(transform, Bijection):
            count.extend(
                self._add_transform(
                    transform.forward,
                    transform.spaces.source,
                    transform.spaces.target,
                    False,
                )
            )
            if invert:
                count.extend(
                    self._add_transform(
                        transform.inverse,
                        transform.spaces.target,
                        transform.spaces.source,
                        False,
                    )
                )
            return count

        src, tgt = self._update_spaces(transform, source, target)

        if self.graph.has_edge(src, tgt):
            logger.warning(f"Replacing existing edge between {src} and {tgt}")

        self.graph.add_edge(src, tgt, transform=transform)
        count.append((src, tgt))
        if invert:
            count.extend(self._add_inverse(transform, src, tgt))
        return count

    def _add_inverse(
        self,
        transform: Transform[ArrayT],
        source: SpaceRef | None,
        target: SpaceRef | None,
    ) -> list[tuple[SpaceRef, SpaceRef]]:
        src, tgt = self._update_spaces(transform, source, target)
        out = []

        if self.graph.has_edge(tgt, src):
            logger.debug(
                "Implicit reverse edge not added to graph as explicit edge already exists for %s->%s",
                tgt,
                src,
            )
        elif t := transform.invert():
            if isinstance(t, Bijection):
                t = t.forward
            self.graph.add_edge(tgt, src, transform=t)
            out.append((tgt, src))
        else:
            logger.debug(
                "Reverse edge not added to graph for non-invertible %s->%s transform",
                src,
                tgt,
            )
        return out

    def add_transform(
        self,
        transform: Transform[ArrayT],
        source: SpaceRef | None = None,
        target: SpaceRef | None = None,
        invert: bool = True,
    ) -> list[tuple[SpaceRef, SpaceRef]]:
        """Add a transform to the graph, optionally with its inverse.

        If the given transform is a `TransformSequence`,
        it will be split down into subsequences where intermediate spaces are known.

        If the given transform is a `Bijection`,
        its forward component will be added as an independent edges;
        if `invert=True`, the same will be done with the inverse component.

        This method will overwrite existing edges.
        Implicit inverses calculated from the given transform will not overwrite existing explicit edges,
        except in the case of the `Bijection`.

        Parameters
        ----------
        transform :
            Transform to add to the graph as an edge.
        source :
            May be omitted if `transform` has its source space defined.
        target : SpaceRef | None, optional
            May be omitted if `transform` has its target space defined.
        invert : bool, optional
            Try to add the reverse edge by inverting the transform if possible; default True

        Returns
        -------
        int
            Number of edges added to the graph.
        """
        out = []
        if isinstance(transform, TransformSequence):
            # TODO: weighting of split-out sequences could be problematic
            ts = split_sequence(transform)
            out.extend(
                chain.from_iterable(
                    self.add_transform(t, None, None, invert) for t in ts
                )
            )

        elif isinstance(transform, Bijection):
            out.extend(
                self.add_transform(
                    transform.forward,
                    source,
                    target,
                    False,
                )
            )
            if invert:
                out.extend(
                    self.add_transform(
                        transform.inverse,
                        target,
                        source,
                        False,
                    )
                )

        else:
            out.extend(self._add_transform(transform, source, target, invert))

        if out:
            self.get_sequence.cache_clear()

        return out

    def add_transforms(
        self,
        transforms: Iterable[Transform[ArrayT]],
        inverse: bool = True,
    ) -> list[tuple[SpaceRef, SpaceRef]]:
        """Bulk-add transformations to the graph.

        Every given transform must have a source and target space defined;
        these spaces are the nodes of the graph.

        This method is preferred over `TransformGraph.add_transform`
        when some reverse edges are explicitly defined
        and you don't want them to be overridden by implicit reverse edges
        when `inverse=True`.

        `Bijection`s and `TransformSequence`s will be split out as documented in
        `TransformGraph.add_transform`.

        Note that a single `TransformSequence` is itself an `Iterable[Transform]`
        and so could be used as the `transforms` argument.
        However, a `TransformSequence` does not require that all of its members
        have explicit source and target spaces,
        where the `transforms` argument here does,
        so not all `TransformSequence`s can be used directly as the argument
        (wrap them in a list instead or use `TransformGraph.add_transform`).

        Parameters
        ----------
        transforms : Iterable[Transform[ArrayT]]
            Transforms which must have a source and target space defined.
        inverse:
            Invert the transformations and add them too.

        Raises
        ------
        ValueError
            Undefined source and target spaces.
        """
        if isinstance(transforms, TransformSequence):
            warnings.warn(
                "add_transforms() argument is a TransformSequence, "
                "which allows undefined intermediate spaces, "
                "in which case this method will fail. "
                "Prefer the add_transform() argument for single logical transforms, "
                "or wrap the given argument in a collection (e.g. a list)."
            )

        forwards = []
        for t in transforms:
            forwards.extend(self.add_transform(t, invert=False))

        if not inverse:
            return forwards

        out = list(forwards)

        # add inverses in second stage to prevent implicit reverse transforms blocking explicit
        for src, tgt in forwards:
            t = self.graph.edges[src, tgt]["transform"]
            out.extend(self._add_inverse(t, src, tgt))

        return out

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
