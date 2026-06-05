from shapely import (
    Point,
    LineString,
    LinearRing,
    Polygon,
    MultiPoint,
    MultiLineString,
    MultiPolygon,
    GeometryCollection,
)

from transformnd.adapters.shapely import GeometryAdapter
from transformnd.transforms import Scale

import pytest


@pytest.mark.parametrize(
    ["original", "expected"],
    [
        (Point(1, 1), Point(2, 3)),
        (
            LineString([[0, 0], [1, 0], [1, 1], [0, 1]]),
            LineString([[0, 0], [2, 0], [2, 3], [0, 3]]),
        ),
        (
            LinearRing([[0, 0], [1, 0], [1, 1], [0, 1]]),
            LinearRing([[0, 0], [2, 0], [2, 3], [0, 3]]),
        ),
        (
            Polygon(
                [[0, 0], [3, 0], [3, 3], [0, 3]], [[[1, 1], [1, 2], [2, 2], [2, 1]]]
            ),
            Polygon(
                [[0, 0], [6, 0], [6, 9], [0, 9]], [[[2, 3], [2, 6], [4, 6], [4, 3]]]
            ),
        ),
        (MultiPoint([[1, 1], [2, 2]]), MultiPoint([[2, 3], [4, 6]])),
        (
            MultiLineString(
                [
                    LineString([[0, 0], [1, 0], [1, 1], [0, 1]]),
                    LineString([[0, 0], [1, 0], [1, 1], [0, 1]]),
                ]
            ),
            MultiLineString(
                [
                    LineString([[0, 0], [2, 0], [2, 3], [0, 3]]),
                    LineString([[0, 0], [2, 0], [2, 3], [0, 3]]),
                ]
            ),
        ),
        (
            MultiPolygon(
                [
                    Polygon(
                        [[0, 0], [3, 0], [3, 3], [0, 3]],
                        [[[1, 1], [1, 2], [2, 2], [2, 1]]],
                    ),
                    Polygon(
                        [[0, 0], [3, 0], [3, 3], [0, 3]],
                        [[[1, 1], [1, 2], [2, 2], [2, 1]]],
                    ),
                ]
            ),
            MultiPolygon(
                [
                    Polygon(
                        [[0, 0], [6, 0], [6, 9], [0, 9]],
                        [[[2, 3], [2, 6], [4, 6], [4, 3]]],
                    ),
                    Polygon(
                        [[0, 0], [6, 0], [6, 9], [0, 9]],
                        [[[2, 3], [2, 6], [4, 6], [4, 3]]],
                    ),
                ]
            ),
        ),
        (
            GeometryCollection(
                [
                    Point(1, 1),
                    LineString([[0, 0], [1, 0], [1, 1], [0, 1]]),
                    LinearRing([[0, 0], [1, 0], [1, 1], [0, 1]]),
                    Polygon(
                        [[0, 0], [3, 0], [3, 3], [0, 3]],
                        [[[1, 1], [1, 2], [2, 2], [2, 1]]],
                    ),
                    MultiPoint([[1, 1], [2, 2]]),
                    MultiLineString(
                        [
                            LineString([[0, 0], [1, 0], [1, 1], [0, 1]]),
                            LineString([[0, 0], [1, 0], [1, 1], [0, 1]]),
                        ]
                    ),
                    MultiPolygon(
                        [
                            Polygon(
                                [[0, 0], [3, 0], [3, 3], [0, 3]],
                                [[[1, 1], [1, 2], [2, 2], [2, 1]]],
                            ),
                            Polygon(
                                [[0, 0], [3, 0], [3, 3], [0, 3]],
                                [[[1, 1], [1, 2], [2, 2], [2, 1]]],
                            ),
                        ]
                    ),
                ]
            ),
            GeometryCollection(
                [
                    Point(2, 3),
                    LineString([[0, 0], [2, 0], [2, 3], [0, 3]]),
                    LinearRing([[0, 0], [2, 0], [2, 3], [0, 3]]),
                    Polygon(
                        [[0, 0], [6, 0], [6, 9], [0, 9]],
                        [[[2, 3], [2, 6], [4, 6], [4, 3]]],
                    ),
                    MultiPoint([[2, 3], [4, 6]]),
                    MultiLineString(
                        [
                            LineString([[0, 0], [2, 0], [2, 3], [0, 3]]),
                            LineString([[0, 0], [2, 0], [2, 3], [0, 3]]),
                        ]
                    ),
                    MultiPolygon(
                        [
                            Polygon(
                                [[0, 0], [6, 0], [6, 9], [0, 9]],
                                [[[2, 3], [2, 6], [4, 6], [4, 3]]],
                            ),
                            Polygon(
                                [[0, 0], [6, 0], [6, 9], [0, 9]],
                                [[[2, 3], [2, 6], [4, 6], [4, 3]]],
                            ),
                        ]
                    ),
                ]
            ),
        ),
    ],
)
def test_geom(original, expected):
    adapter = GeometryAdapter()
    transform = Scale([2, 3])
    out = adapter.apply(transform, original)

    assert out == expected
