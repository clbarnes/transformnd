"""
.. include:: ../../README.md

You can find some examples here:

- [Tutorial](./examples/tutorial.html)
- [Image transformation](./examples/image.html)

"""

from .base import Transform, TransformSequence, TransformFnWrapper
from .types import TransformSignature, NDims, SpaceRef
from . import transforms
from . import adapters
from .graph import TransformGraph
from .spaced import Spaced
from importlib.metadata import version as _version

__version__ = _version("transformnd")

__all__ = [
    "Transform",
    "TransformGraph",
    "TransformSequence",
    "TransformFnWrapper",
    "TransformSignature",
    "SpaceRef",
    "transforms",
    "adapters",
    "NDims",
    "Spaced",
]
