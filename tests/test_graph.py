from copy import deepcopy

from transformnd.base import Transform, TransformSequence
from transformnd.graph import TransformGraph
from transformnd.transforms.simple import Translate, Scale
from transformnd.transforms.affine import Affine
from transformnd import Spaced

import pytest

import numpy as np

from transformnd.util import as_floats


def test_add_transforms():
    t: TransformGraph = TransformGraph()
    t.add_transform(Spaced(Scale([2, 2]), 1, 2))
    t.add_transform(Spaced(Translate([10, 20]), 2, 3))


@pytest.mark.parametrize(("full",), [(True,), (False,)])
def test_noop_path(coords5x3, full):
    t: TransformGraph = TransformGraph()
    t.add_transform(Spaced(Scale([2, 2]), 1, 2))
    s = t.get_sequence(1, 1, full=full)
    assert s.apply(coords5x3) == pytest.approx(coords5x3)


def test_path():
    t1 = Scale([2, 3])
    t2 = Translate([10, 20])
    transforms = [t1, t2]
    g: TransformGraph = TransformGraph()
    g.add_transform(Spaced(t1, 1, 2))
    g.add_transform(Spaced(t2, 2, 3))
    spaces = (1, 3)
    seq = g.get_sequence(*spaces, full=True)
    assert len(seq) == 2
    assert isinstance(seq[0], Scale)
    assert isinstance(seq[1], Translate)

    in_coords = as_floats([[0, 0], [1, 1], [2, 2]])
    res = g.transform(1, 3, in_coords)
    expected = TransformSequence(transforms).apply(in_coords)
    assert res == pytest.approx(expected)


def test_graph_traversal():
    graph: TransformGraph = TransformGraph()
    graph.add_transform(Spaced(Translate([1, 2]), 1, 2))
    graph.add_transform(Spaced(Scale([2, 3]), 2, 3))
    graph.add_transform(Spaced(Affine(np.eye(3)), 3, 4))

    seq = graph.get_sequence(1, 4)
    assert isinstance(seq, TransformSequence)
    assert len(seq) == 1

    simplified_seq = seq.simplify()

    expected_affine = np.array([[2, 0, 2], [0, 3, 6], [0, 0, 1]])
    got_affine = simplified_seq.to_affine()
    assert got_affine is not None
    assert got_affine.matrix == pytest.approx(expected_affine)


def test_multigraph():
    src, tgt = 0, 1
    orig = Translate([0, 1])
    other = Scale([0, 2])
    n = 10
    transforms: list[Transform] = [deepcopy(other) for _ in range(n)]
    transforms.append(orig)
    transforms.extend(deepcopy(other) for _ in range(n))

    tgraph: TransformGraph = TransformGraph()
    for t in transforms:
        if t is orig:
            weight = 1
        else:
            weight = 2
        tgraph.add_transform(Spaced(t, src, tgt), edge_data={"weight": weight})

    assert len(tgraph.graph.edges) == len(transforms)
    seq = tgraph.get_sequence(src, tgt, full=True, weight="weight")
    assert len(seq) == 1
    assert isinstance(seq[0], Translate)


if __name__ == "__main__":
    test_graph_traversal()
    print("All tests passed!")
