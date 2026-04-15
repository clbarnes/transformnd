import numpy as np

from transformnd.transforms import MapAxis


def test_2d_map_axis():
    # 1. Test apply
    map = MapAxis(permutation=[1, 0])
    a = np.array([[1, 2], [3, 4]])
    a_mapped = np.array([[2, 1], [4, 3]])
    assert np.array_equal(map.apply(a), a_mapped)
    # 2. Test invert
    map_inv = ~map
    assert np.array_equal(map_inv.apply(a_mapped), a)


def test_3d_map_axis():
    map = MapAxis(permutation=[2, 0, 1])
    a = np.array([[1, 2, 3], [4, 5, 6]])
    a_mapped = np.array([[3, 1, 2], [6, 4, 5]])
    assert np.array_equal(map.apply(a), a_mapped)
    # 2. Test invert
    map_inv = ~map
    assert np.array_equal(map_inv.apply(a_mapped), a)
