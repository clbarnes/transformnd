import numpy as np

from transformnd.transforms.simple import Identity, Scale, Translate
from transformnd.types import Spaces


def test_identity_spaces():
    t = Identity[np.ndarray](1, spaces=Spaces(1, 1))
    assert t.spaces.target == 1


def test_translate_nd(coords5x3):
    t = Translate[np.ndarray](1)
    assert np.allclose(t.apply(coords5x3), coords5x3 + 1)


def test_translate_3d(coords5x3):
    trans = [1, 2, 3]
    t = Translate[np.ndarray](np.array(trans))
    assert np.allclose(t.apply(coords5x3), coords5x3 + trans)


def test_translate_neg(coords5x3):
    t_neg = ~Translate(1)
    assert np.allclose(t_neg.apply(coords5x3), coords5x3 - 1)


def test_scale_nd(coords5x3):
    t = Scale[np.ndarray](2)
    assert np.allclose(t.apply(coords5x3), coords5x3 * 2)


def test_scale_3d(coords5x3):
    scale = [2, 3, 4]
    t = Scale[np.ndarray](np.array(scale))
    assert np.allclose(t.apply(coords5x3), coords5x3 * scale)


def test_scale_neg(coords5x3):
    t_neg = ~Scale(2)
    assert np.allclose(t_neg.apply(coords5x3), coords5x3 / 2)
