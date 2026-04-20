import numpy as np

from transformnd.extents.bounding_box import BoundingBox
from transformnd.adapters.bounding_box import BoundingBoxAdapter
from transformnd.transforms import Scale

import pytest


def test_bbox_adapter():
    bbox = BoundingBox(np.array([1, 1], float), np.array([2, 2], float))
    t = Scale(np.array([10, 10], float))
    a = BoundingBoxAdapter()
    bb2 = a.apply(t, bbox)
    assert bb2.mins == pytest.approx(np.array([10, 10], float))
    assert bb2.maxes == pytest.approx(np.array([20, 20], float))
