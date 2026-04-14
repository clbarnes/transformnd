"""
.. include:: ../../README.md
"""

from .base import Transform, TransformSequence, TransformWrapper
from .util import SpaceRef, TransformSignature, check_ndim
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
    "check_ndim",
    "transforms",
    "adapters",
]
