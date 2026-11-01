"""
.. include:: ../../README.md

You can find some examples here:

- [Tutorial](./examples/tutorial.html)
- [Image transformation](./examples/image.html)

"""

from importlib.metadata import version as _version

from . import adapters, transforms
from .base import Transform, TransformFnWrapper, TransformSequence
from .graph import TransformGraph
from .spaced import Spaced
from .types import NDims, SpaceRef, TransformSignature

__version__ = _version("transformnd")

__all__ = [
    "NDims",
    "SpaceRef",
    "Spaced",
    "Transform",
    "TransformFnWrapper",
    "TransformGraph",
    "TransformSequence",
    "TransformSignature",
    "adapters",
    "transforms",
]
