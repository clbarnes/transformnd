"""
.. include:: ../../README.md

You can find some examples here:

- [Tutorial](./examples/tutorial.html)
- [Image transformation](./examples/image.html)

"""

from .base import Transform, TransformSequence, TransformWrapper
from .util import SpaceRef
from .types import Spaces, TransformSignature, NDims
from . import transforms
from . import adapters
from .graph import TransformGraph
from importlib.metadata import version as _version

__version__ = _version("transformnd")

__all__ = [
    "Transform",
    "TransformGraph",
    "TransformSequence",
    "TransformWrapper",
    "TransformSignature",
    "SpaceRef",
    "transforms",
    "adapters",
    "Spaces",
    "NDims",
]
