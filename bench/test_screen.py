import numpy as np
import pytest

from transformnd.transforms.simple import Identity, Translate, Scale, Transform
from transformnd.transforms.affine import Affine


@pytest.mark.parametrize(
    "transform",
    [
        Identity(2),
        Translate([1, 2]),
        Scale([2, 3]),
        Affine([[2, 1, 1], [0, 1, 0], [0, 0, 1]]),
    ],
    ids=lambda t: type(t).__qualname__,
)
def test_screen(transform: Transform, screen_coords: np.ndarray, benchmark):
    def fn():
        transform.apply(screen_coords)

    benchmark(fn)
