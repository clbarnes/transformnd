from random import Random
from transformnd.transforms import ProjectAxis
from transformnd.util import as_floats
import pytest


def test_insert():
    t = ProjectAxis(created={0}, source_ndim=2, target_ndim=3)

    out = t.apply(as_floats([[1, 2], [3, 4]]))
    assert out == pytest.approx(as_floats([[0, 1, 2], [0, 3, 4]]))


def test_remove():
    t = ProjectAxis(dropped={0}, source_ndim=3, target_ndim=2)
    out = t.apply(as_floats([[1, 2, 3], [4, 5, 6]]))
    assert out == pytest.approx(as_floats([[2, 3], [5, 6]]))


def test_invert():
    t = ProjectAxis({0}, {2}, 3, 3)
    ti = t.invert()
    assert ti is not None
    assert ti.dropped == {2}
    assert ti.created == {0}


def random_ops(in_ndim: int, out_ndim: int, seed=1991) -> ProjectAxis:
    rng = Random(seed)
    n_dropped = rng.randint(max(0, in_ndim - out_ndim), in_ndim - 1)
    to_drop = list(range(in_ndim))
    rng.shuffle(to_drop)
    dropped = set(to_drop[:n_dropped])

    n_created = out_ndim - (in_ndim - n_dropped)
    to_create = list(range(out_ndim))
    rng.shuffle(to_create)
    created = set(to_create[:n_created])

    return ProjectAxis(dropped, created, in_ndim, out_ndim)


def test_to_affine(coords5x3):
    t = random_ops(coords5x3.shape[-1], 3)
    aff = t.to_affine()
    assert aff is not None
    assert aff.ndims == t.ndims
    assert t.apply(coords5x3) == pytest.approx(aff.apply(coords5x3))
