import numpy as np
from shapely.geometry import (
    LinearRing,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry

from ..base import Transform
from .base import BaseAdapter


class GeometryAdapter(BaseAdapter[BaseGeometry]):
    """Transform shapely geometries.

    As well as the generic `__call__()`,
    there are methods for transforming different geometry subclasses.

    N.B. some transforms may create invalid topologies
    (incorrect winding, self-intersections etc.)
    """

    def transform_point(self, transform: Transform, point: Point) -> Point:
        return Point(*transform.apply(np.array(point.coords))[0])

    def transform_linestring(
        self, transform: Transform, linestring: LineString
    ) -> LineString:
        return LineString(transform.apply(np.array(linestring.coords)))

    def transform_linear_ring(
        self, transform: Transform, linear_ring: LinearRing
    ) -> LinearRing:
        return LinearRing(transform.apply(np.array(linear_ring.coords)))

    def transform_polygon(self, transform: Transform, polygon: Polygon) -> Polygon:
        return Polygon(
            self.transform_linear_ring(transform, polygon.exterior),
            [self.transform_linear_ring(transform, i) for i in polygon.interiors],
        )

    def transform_multi(self, transform: Transform, multi_geom):
        cls = type(multi_geom)
        method = {
            MultiPoint: self.transform_point,
            MultiLineString: self.transform_linestring,
            MultiPolygon: self.transform_polygon,
        }[cls]
        return cls([method(transform, obj) for obj in multi_geom])

    def apply(
        self,
        transform: Transform,
        obj: BaseGeometry,
    ) -> BaseGeometry:
        """Transform the shapely geometry.

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
            return self.transform_multi(transform, obj)

        method = {
            Point: self.transform_point,
            LineString: self.transform_linestring,
            LinearRing: self.transform_linear_ring,
            Polygon: self.transform_polygon,
        }[type(obj)]
        return method(transform, obj)
