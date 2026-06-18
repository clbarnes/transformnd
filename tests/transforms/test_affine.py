import numpy as np
import pytest

from transformnd.base import TransformSequence
from transformnd.transforms.affine import Affine
from transformnd.transforms.simple import Scale, Translate
from transformnd.util import as_floats


def test_identity():
    i2 = Affine.identity(2)
    test = np.array([(1, 1), (2, 3), (-2, 50)], float)
    ref = test.copy()
    assert np.allclose(i2.apply(test), ref)
    assert np.allclose((~i2).apply(test), ref)


@pytest.mark.parametrize(["ndim"], [[d] for d in range(1, 6)])
def test_translation(ndim, rng):
    t_arr = np.arange(ndim) + 1

    coords = rng.random((5, ndim)) - 0.5
    trans_arr = Affine.translation(t_arr)
    assert np.allclose(trans_arr.apply(coords), coords + t_arr)
    assert np.allclose((~trans_arr).apply(coords), coords - t_arr)


@pytest.mark.parametrize(["ndim"], [[d] for d in range(2, 6)])
def test_scaling(ndim, rng):
    s_arr = np.arange(ndim) + 2

    coords = rng.random((5, ndim)) - 0.5
    trans = Affine.scaling(s_arr)
    assert np.allclose(trans.apply(coords), coords * s_arr)
    assert np.allclose((~trans).apply(coords), coords / s_arr)

    trans_arr = Affine.scaling(s_arr)
    assert np.allclose(trans_arr.apply(coords), coords * s_arr)


def test_rotation2():
    rot90 = Affine.rotation2(90)
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


def test_downprojection():
    lin_map = as_floats(
        [
            [1, 0, 0],
            [0, 1, 0],
        ]
    )
    t = Affine.from_linear_map(lin_map)
    assert t.ndims.source == 3
    assert t.ndims.target == 2

    coords = as_floats([[1, 2, 3], [4, 5, 6]])
    out = t.apply(coords)
    assert out == pytest.approx(as_floats([[1, 2], [4, 5]]))


def test_upprojection():
    lin_map = as_floats(
        [
            [1, 0],
            [0, 1],
            [0, 0],
        ]
    )
    t = Affine.from_linear_map(lin_map)
    assert t.ndims.source == 2
    assert t.ndims.target == 3

    coords = as_floats([[1, 2], [3, 4], [5, 6]])
    out = t.apply(coords)
    assert out == pytest.approx(as_floats([[1, 2, 0], [3, 4, 0], [5, 6, 0]]))


def test_transpose_commutation():
    mx_dim = 10

    rng = np.random.default_rng(1991)
    for _ in range(100):
        lhs_shape = rng.integers(1, mx_dim, 2, endpoint=True)
        lhs = rng.random(tuple(lhs_shape))
        rhs_shape = (lhs_shape[1], rng.integers(mx_dim, endpoint=True))
        rhs = rng.random(rhs_shape)

        lr = lhs @ rhs
        rtlt_t = (rhs.T @ lhs.T).T

        assert lr == pytest.approx(rtlt_t)


# def test_reflection():
#     pass


# def test_rotation3():
#     pass


# def test_shear():
#     pass
