"""Adapters for transforming objects which are not well-behaved numpy arrays."""
from .base import Adapter, AttrAdapter, BaseAdapter, FnAdapter

__all__ = ["BaseAdapter", "Adapter", "FnAdapter", "AttrAdapter"]
