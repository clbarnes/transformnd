"""
.. include:: ../../README.md
"""
from .base import Transform, TransformSequence, TransformWrapper
from .util import SpaceRef, TransformSignature, check_ndim
from .version import version as __version__  # noqa: F401
from .version import version_tuple as __version_info__  # noqa: F401

__all__ = [
    "Transform",
    "TransformSequence",
    "TransformWrapper",
    "TransformSignature",
    "SpaceRef",
    "check_ndim",
]
