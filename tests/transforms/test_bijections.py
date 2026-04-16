import numpy as np
from transformnd.transforms import Bijection, Scale, MapAxis

import pytest

@pytest.mark.parametrize("scale", [1, 2, 3])
def test_bijection_apply_scale(scale):
    coords5x3 = np.random.rand(5, 3)
    forward = Scale(scale)
    inverse = Scale(1 / scale)
    bij = Bijection(forward, inverse)
    assert np.allclose(bij.apply(coords5x3), coords5x3 * scale)


def test_bijection_apply_map_axis():
    coords2x2 = np.array([[1, 2], [3, 4]])
    forward = MapAxis(permutation=[1, 0])
    inverse = MapAxis(permutation=[1, 0])
    bij = Bijection(forward, inverse)
    coords2x2_inv = bij.apply(coords2x2)
    assert np.array_equal(coords2x2_inv, np.array([[2, 1], [4, 3]]))
    bij_inv = ~bij
    assert np.array_equal(bij_inv.apply(coords2x2_inv), coords2x2)


def test_bijection_invert_scale():
    coords5x3 = np.random.rand(5, 3)
    forward = Scale(2)
    inverse = Scale(0.5)
    bij = Bijection(forward, inverse)
    bij_inv = ~bij
    assert np.allclose(bij_inv.apply(coords5x3), coords5x3 * 0.5)

def test_bijection_invert_map_axis():
    # [2, 0, 1] is the inverse of [1, 2, 0], not its own inverse
    coords3x2 = np.array([[1, 2, 3], [4, 5, 6]])
    forward = MapAxis(permutation=[1, 2, 0])
    inverse = MapAxis(permutation=[2, 0, 1])
    bij = Bijection(forward, inverse)
    coords3x2_mapped = bij.apply(coords3x2)
    coords3x2_inv = (~bij).apply(coords3x2_mapped)
    assert np.array_equal(coords3x2_inv, coords3x2)


def test_bijection_roundtrip(coords5x3):
    forward = Scale(3)
    inverse = Scale(1 / 3)
    bij = Bijection(forward, inverse)
    assert np.allclose((~bij).apply(bij.apply(coords5x3)), coords5x3)
