import numpy as np

from transformnd.transforms.simple import Identity, Scale, Translate
from transformnd.types import Spaces
import pytest


def test_identity_spaces():
    t = Identity(1, spaces=Spaces(1, 1))
    assert t.spaces.target == 1


def test_translate_3d(coords5x3):
    trans = [1, 2, 3]
    t = Translate(np.array(trans))
    assert np.allclose(t.apply(coords5x3), coords5x3 + trans)


def test_translate_neg(coords5x3):
    t_neg = ~Translate([1] * 3)
    assert np.allclose(t_neg.apply(coords5x3), coords5x3 - 1)


def test_scale_3d(coords5x3):
    scale = [2, 3, 4]
    t = Scale(np.array(scale))
    assert np.allclose(t.apply(coords5x3), coords5x3 * scale)


def test_scale_neg(coords5x3):
    t_neg = ~Scale([2] * 3)
    assert np.allclose(t_neg.apply(coords5x3), coords5x3 / 2)


def test_translation_jax(coords5x3):
    import jax.numpy as jnp

    t = Translate([1, 2, 3])
    expected = t.apply(coords5x3)
    t2 = t.to_device(jnp)
    c2 = jnp.asarray(coords5x3)

    out = t2.apply(c2)  # type:ignore
    assert out == pytest.approx(expected)


def test_scale_jax(coords5x3):
    import jax.numpy as jnp

    t = Scale([1, 2, 3])
    expected = t.apply(coords5x3)
    t2 = t.to_device(jnp)
    c2 = jnp.asarray(coords5x3)

    out = t2.apply(c2)  # type:ignore
    assert out == pytest.approx(expected)
