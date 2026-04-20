import numpy as np
from typing import Callable
from shapely.geometry import (
    LinearRing,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    GeometryCollection,
)
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry
from shapely.coords import CoordinateSequence

from ..base import Transform, ArrayT
from .base import BaseAdapter


def as_numpy(coords: CoordinateSequence) -> np.ndarray:
    return np.asarray(coords)


class GeometryAdapter(BaseAdapter[BaseGeometry, ArrayT]):
    """Transform shapely geometries.

    As well as the generic `apply()`,
    there are `apply_*()` methods for transforming different geometry subclasses.

    N.B. some transforms may create invalid topologies
    (incorrect winding, self-intersections etc.)
    """

    def __init__(
        self, array_fn: Callable[[CoordinateSequence], ArrayT] = as_numpy
    ) -> None:
        self.array_fn = array_fn

    def apply_point(self, transform: Transform, point: Point) -> Point:
        return Point(*transform.apply(self.array_fn(point.coords))[0])

    def apply_linestring(
        self, transform: Transform, linestring: LineString
    ) -> LineString:
        return LineString(transform.apply(self.array_fn(linestring.coords)))

    def apply_linear_ring(
        self, transform: Transform, linear_ring: LinearRing
    ) -> LinearRing:
        return LinearRing(transform.apply(self.array_fn(linear_ring.coords)))

    def apply_polygon(self, transform: Transform, polygon: Polygon) -> Polygon:
        return Polygon(
            self.apply_linear_ring(transform, polygon.exterior),
            [self.apply_linear_ring(transform, i) for i in polygon.interiors],
        )

    def apply_multipoint(self, transform: Transform, obj: MultiPoint):
        return MultiPoint([self.apply_point(transform, o) for o in obj.geoms])

    def apply_multilinestring(self, transform: Transform, obj: MultiLineString):
        return MultiLineString([self.apply_linestring(transform, o) for o in obj.geoms])

    def apply_multipolygon(self, transform: Transform, obj: MultiPolygon):
        return MultiPolygon([self.apply_polygon(transform, o) for o in obj.geoms])

    def apply_multipart(
        self, transform: Transform, obj: BaseMultipartGeometry
    ) -> BaseMultipartGeometry:
        """Apply the transform to any shapely multipart geometry."""
        if isinstance(obj, MultiPoint):
            return self.apply_multipoint(transform, obj)
        elif isinstance(obj, MultiLineString):
            return self.apply_multilinestring(transform, obj)
        elif isinstance(obj, MultiPolygon):
            return self.apply_multipolygon(transform, obj)
        elif isinstance(obj, GeometryCollection):
            return self.apply_collection(transform, obj)
        else:
            raise ValueError(f"Unknown multipart geometry type {type(obj)}")

    def apply_collection(self, transform: Transform, obj: GeometryCollection):
        return GeometryCollection([self.apply(transform, o) for o in obj.geoms])

    def apply(
        self,
        transform: Transform,
        obj: BaseGeometry,
    ) -> BaseGeometry:
        """Transform the shapely geometry.

        See the other `apply_*` methods if you already know what type of geometry you're working with;
        this may be a bit faster.

        Parameters
        ----------
        transform : Transform
        obj : BaseGeometry
            Some shapely geometry in 2 or 3D

        Returns
        -------
        BaseGeometry
            An object of the same type as the input.
        """
        if isinstance(obj, BaseMultipartGeometry):
            return self.apply_multipart(transform, obj)
        elif isinstance(obj, Point):
            return self.apply_point(transform, obj)
        elif isinstance(obj, LineString):
            return self.apply_linestring(transform, obj)
        elif isinstance(obj, LinearRing):
            return self.apply_linear_ring(transform, obj)
        elif isinstance(obj, Polygon):
            return self.apply_polygon(transform, obj)
        else:
            raise ValueError(f"Unknown geometry type {type(obj)}")
