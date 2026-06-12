import numpy as np
import pytest

SEED = 1991


@pytest.fixture
def rng() -> np.random.RandomState:
    return np.random.RandomState(SEED)


@pytest.fixture
def screen_coords() -> np.ndarray:
    y = np.arange(1080, dtype="float32")
    x = np.arange(1920, dtype="float32")
    return np.dstack(np.meshgrid(y, x)).reshape((-1, 2))
