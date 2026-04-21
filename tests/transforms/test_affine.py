import numpy as np
import pytest

from transformnd.base import TransformSequence
from transformnd.transforms.affine import Affine
from transformnd.transforms.simple import Scale, Translate


def test_identity():
    i2 = Affine[np.ndarray].identity(2)
    test = np.array([(1, 1), (2, 3), (-2, 50)], float)
    ref = test.copy()
    assert np.allclose(i2.apply(test), ref)
    assert np.allclose((~i2).apply(test), ref)


@pytest.mark.parametrize(["ndim"], [[d] for d in range(1, 6)])
def test_translation(ndim, rng):
    t = 1

    coords = rng.random((5, ndim)) - 0.5
    t_arr = [t] * ndim
    trans = Affine[np.ndarray].translation(t, ndim)
    trans_arr = Affine[np.ndarray].translation(t_arr)
    assert np.allclose(trans.apply(coords), coords + t)
    assert np.allclose(trans_arr.apply(coords), coords + t)
    assert np.allclose((~trans).apply(coords), coords - t)


@pytest.mark.parametrize(["ndim"], [[d] for d in range(2, 6)])
def test_scaling(ndim, rng):
    s = 2

    coords = rng.random((5, ndim)) - 0.5
    trans = Affine[np.ndarray].scaling(s, ndim)
    assert np.allclose(trans.apply(coords), coords * s)
    assert np.allclose((~trans).apply(coords), coords / s)

    t_arr = [s] * ndim
    trans_arr = Affine[np.ndarray].scaling(t_arr)
    assert np.allclose(trans_arr.apply(coords), coords * s)


def test_rotation2():
    rot90 = Affine[np.ndarray].rotation2(90)
    coords = np.array([[1, 1]])
    expected = np.array([[-1, 1], [-1, -1], [1, -1], [1, 1]])
    for exp in expected:
        coords = rot90.apply(coords)
        assert np.allclose(coords[0], exp)
    inv = ~rot90
    for exp in reversed(expected[:-1]):
        coords = inv.apply(coords)
        assert np.allclose(coords[0], exp)


def test_matmul(subtests):
    # scale factors for 2D on the first diagonal;
    # bottom right must be 1, otherwise bottom row must be 0
    scale = np.array([[2, 0, 0], [0, 3, 0], [0, 0, 1]], float)
    # 3rd column is all 1s to fit the affine matrix
    coords = np.array([[1, 2, 1], [3, 4, 1]], float)
    s_result = coords @ scale.T
    s_expected = np.array(
        [
            [2, 6, 1],
            [6, 12, 1],
        ],
        float,
    )
    with subtests.test(msg="scale results"):
        assert s_result == pytest.approx(s_expected)

    translation = np.array([[1, 0, 10], [0, 1, 20], [0, 0, 1]], float)
    t_result = coords @ translation.T
    t_expected = np.array(
        [
            [11, 22, 1],
            [13, 24, 1],
        ],
        float,
    )
    with subtests.test(msg="translation results"):
        assert t_result == pytest.approx(t_expected)

    scale_translate = translation @ scale
    expected_scale_translation = np.array(
        [
            [2, 0, 10],
            [0, 3, 20],
            [0, 0, 1],
        ],
        float,
    )
    with subtests.test(msg="scale->translate matrix"):
        assert scale_translate == pytest.approx(expected_scale_translation)

    with subtests.test(msg="scale->translate results"):
        expected = (coords @ scale.T) @ translation.T
        assert coords @ scale_translate.T == pytest.approx(expected)

    translation_scale = scale @ translation
    with subtests.test(msg="translate->scale results"):
        expected = (coords @ translation.T) @ scale.T
        assert coords @ translation_scale.T == pytest.approx(expected)


def test_affine_combination(rng):
    scale = Affine.scaling([1.5, 2.5])
    translate = Affine.translation([11.5, 22.5])

    inputs = rng.random((5, 2))

    st_seq = TransformSequence([scale, translate])
    st = translate @ scale
    assert st.apply(inputs) == pytest.approx(st_seq.apply(inputs))

    ts_seq = TransformSequence([translate, scale])
    ts = scale @ translate
    assert ts.apply(inputs) == pytest.approx(ts_seq.apply(inputs))


def test_inversion(rng):
    scale = Scale([1.5, 2.5])
    translate = Translate([11.5, 22.5])

    scale_aff = scale.to_affine()
    assert scale_aff is not None
    translate_aff = translate.to_affine()
    assert translate_aff is not None

    aff = translate_aff @ scale_aff
    inv_aff = ~aff

    coords = rng.random((5, 2))

    assert inv_aff.apply(aff.apply(coords)) == pytest.approx(coords)


# def test_reflection():
#     pass


# def test_rotation3():
#     pass


# def test_shear():
#     pass
