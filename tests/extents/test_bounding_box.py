from transformnd.extents.bounding_box import BoundingBox
import numpy as np


def test_bbox():
    coords = np.array([[0], [1], [2], [3], [4]], dtype=float)
    bbox = BoundingBox(np.array([1]), np.array([3]))
    res = bbox.contains(coords)
    assert np.array_equal(res, [[False], [True], [True], [False], [False]])
