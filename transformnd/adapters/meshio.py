from meshio import Mesh

from .base import AttrAdapter


class MeshAdapter(AttrAdapter[Mesh]):
    """Transform meshio.Mesh objects.

    N.B. Some transformations may create invalid meshes
    (incorrect winding, self-intersections, inversions etc.).
    """

    def __init__(self):
        super().__init__(points=None)
