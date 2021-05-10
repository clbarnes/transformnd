import sys
import numpy as np
import pytest

sys.path.pop(0)


def make_coords(shape):
    return np.arange(np.product(shape)).reshape(shape)


@pytest.fixture
def coords5x3():
    return make_coords((5, 3))
