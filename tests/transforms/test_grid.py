from transformnd.transforms import GridInterpolation
import pytest
import numpy as np


def test_pow(coords5x3):
    t = GridInterpolation([lambda x: x**2, lambda x: x**3, lambda x: x**4])
    out = t.apply(coords5x3)
    assert out == pytest.approx(np.pow(coords5x3, [2, 3, 4]))
