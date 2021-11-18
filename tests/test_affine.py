import numpy as np
import pytest

from transformnd.transforms.affine import AffineTransform


def test_identity():
    i2 = AffineTransform.identity(2)
    test = np.array([(1, 1), (2, 3), (-2, 50)])
    ref = test.copy()
    assert np.allclose(i2(test), ref)
    assert np.allclose((-i2)(test), ref)


@pytest.mark.parametrize(["ndim"], [[d] for d in range(2, 6)])
def test_translation(ndim, rng):
    t = 1

    coords = rng.random((5, ndim)) - 0.5
    t_arr = [t] * ndim
    trans = AffineTransform.translation(t, ndim)
    trans_arr = AffineTransform.translation(t_arr)
    assert np.allclose(trans(coords), coords + t)
    assert np.allclose(trans_arr(coords), coords + t)
    assert np.allclose((-trans)(coords), coords - t)


@pytest.mark.parametrize(["ndim"], [[d] for d in range(2, 6)])
def test_scaling(ndim, rng):
    s = 2

    coords = rng.random((5, ndim)) - 0.5
    trans = AffineTransform.scaling(s, ndim)
    assert np.allclose(trans(coords), coords * s)
    assert np.allclose((-trans)(coords), coords / s)

    t_arr = [s] * ndim
    trans_arr = AffineTransform.scaling(t_arr)
    assert np.allclose(trans_arr(coords), coords * s)


def test_rotation2():
    rot90 = AffineTransform.rotation2(90)
    coords = np.array([[1, 1]])
    expected = np.array([[-1, 1], [-1, -1], [1, -1], [1, 1]])
    for exp in expected:
        coords = rot90(coords)
        assert np.allclose(coords[0], exp)
    inv = -rot90
    for exp in reversed(expected[:-1]):
        coords = inv(coords)
        assert np.allclose(coords[0], exp)


# def test_reflection():
#     pass


# def test_rotation3():
#     pass


# def test_shear():
#     pass
