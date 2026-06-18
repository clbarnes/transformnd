from random import Random
from transformnd.transforms import ProjectAxis
from transformnd.transforms.project_axis import Insert, Remove
from transformnd.util import as_floats
import pytest


def test_insert():
    t = ProjectAxis([Insert(0)], 2, 3)

    out = t.apply(as_floats([[1, 2], [3, 4]]))
    assert out == pytest.approx(as_floats([[0, 1, 2], [0, 3, 4]]))


def test_remove():
    t = ProjectAxis([Remove(0)], 3, 2)
    out = t.apply(as_floats([[1, 2, 3], [4, 5, 6]]))
    assert out == pytest.approx(as_floats([[2, 3], [5, 6]]))


def test_invert():
    t = ProjectAxis([Remove(0), Insert(2)], 3, 3)
    ti = t.invert()
    assert ti is not None
    assert ti.operations == [Remove(2), Insert(0)]


def random_ops(ndim: int, n_ops: int, seed=1991) -> ProjectAxis:
    rng = Random(seed)
    ops = []
    source_ndim = ndim
    for _ in range(n_ops):
        if ndim == 0:
            Cls = Insert
        else:
            Cls = rng.choice([Insert, Remove])
        if Cls == Remove:
            mx = ndim - 1
        else:
            mx = ndim
        idx = rng.randint(0, mx)
        op = Cls(idx)
        ndim = op.check(ndim)
        ops.append(Cls(idx))

    return ProjectAxis(ops, source_ndim)


def test_to_affine(coords5x3):
    t = random_ops(coords5x3.shape[-1], 3)
    aff = t.to_affine()
    assert aff is not None
    assert aff.ndims == t.ndims
    assert t.apply(coords5x3) == pytest.approx(aff.apply(coords5x3))
