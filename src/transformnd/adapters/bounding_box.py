from array_api_compat import array_namespace
from transformnd.base import Transform
from .base import BaseAdapter
from ..extents.bounding_box import BoundingBox


class BoundingBoxAdapter[ArrayT](BaseAdapter[BoundingBox]):
    def apply(
        self, transform: Transform, obj: BoundingBox[ArrayT]
    ) -> BoundingBox[ArrayT]:
        xp = array_namespace(obj.mins)
        stacked = xp.stack([obj.mins, obj.maxes])
        transformed = transform.apply(stacked)
        mins, maxes = xp.unstack(transformed)
        return BoundingBox(mins, maxes)
