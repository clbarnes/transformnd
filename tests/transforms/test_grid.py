from transformnd.transforms import GridInterpolation
import pytest


def test_simple(coords5x3):
    t = GridInterpolation([lambda x: x * 2, lambda x: x * 3, lambda x: x * 4])
    out = t.apply(coords5x3)
    assert out == pytest.approx(coords5x3 * [2, 3, 4])
