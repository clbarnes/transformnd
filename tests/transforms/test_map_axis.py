import pytest

from transformnd.transforms import MapAxis
from transformnd.util import as_floats


def test_2d_map_axis():
    # 1. Test apply
    map = MapAxis(permutation=[1, 0])
    a = as_floats([[1, 2], [3, 4]])
    a_mapped = as_floats([[2, 1], [4, 3]])
    assert map.apply(a) == pytest.approx(a_mapped)
    # 2. Test invert
    map_inv = ~map
    assert map_inv.apply(a_mapped) == pytest.approx(a)


def test_3d_map_axis():
    map = MapAxis(permutation=[2, 0, 1])
    a = as_floats([[1, 2, 3], [4, 5, 6]])
    a_mapped = as_floats([[3, 1, 2], [6, 4, 5]])
    assert map.apply(a) == pytest.approx(a_mapped)
    # 2. Test invert
    map_inv = ~map
    assert map_inv.apply(a_mapped) == pytest.approx(a)


def test_to_affine():
    map = MapAxis(permutation=[1, 0])
    a = as_floats([[1, 2], [3, 4]])
    a_mapped = as_floats([[2, 1], [4, 3]])
    aff = map.to_affine()
    assert aff is not None
    assert aff.apply(a) == pytest.approx(a_mapped)
