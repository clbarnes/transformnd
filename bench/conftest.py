import numpy as np
import pytest

SEED = 1991


@pytest.fixture
def rng() -> np.random.RandomState:
    return np.random.RandomState(SEED)
