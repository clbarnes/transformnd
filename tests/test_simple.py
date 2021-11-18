import numpy as np
import pytest

from transformnd.transforms.simple import IdentityTransform, Scale, Translate


def test_identity_spaces():
    t = IdentityTransform(source_space=1)
    assert t.target_space == 1

    with pytest.raises(ValueError):
        IdentityTransform(source_space=1, target_space=2)


def test_translate_nd(coords5x3):
    t = Translate(1)
    assert np.allclose(t(coords5x3), coords5x3 + 1)


def test_translate_3d(coords5x3):
    trans = [1, 2, 3]
    t = Translate(np.array(trans))
    assert np.allclose(t(coords5x3), coords5x3 + trans)


def test_translate_neg(coords5x3):
    t_neg = -Translate(1)
    assert np.allclose(t_neg(coords5x3), coords5x3 - 1)


def test_scale_nd(coords5x3):
    t = Scale(2)
    assert np.allclose(t(coords5x3), coords5x3 * 2)


def test_scale_3d(coords5x3):
    scale = [2, 3, 4]
    t = Scale(np.array(scale))
    assert np.allclose(t(coords5x3), coords5x3 * scale)


def test_scale_neg(coords5x3):
    t_neg = -Scale(2)
    assert np.allclose(t_neg(coords5x3), coords5x3 / 2)
