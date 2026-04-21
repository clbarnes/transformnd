import numpy as np
import pytest

SEED = 1991


def make_coords(shape):
    return np.arange(np.prod(shape)).reshape(shape)


@pytest.fixture
def coords5x3() -> np.ndarray:
    return make_coords((5, 3))


@pytest.fixture
def rng() -> np.random.RandomState:
    return np.random.RandomState(SEED)
