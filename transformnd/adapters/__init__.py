"""Adapters for transforming objects which are not well-behaved numpy arrays.

Adapter instances are callables which take the transform to be applied,
the object to apply it to, and optionally some other arguments.
The adapter knows how to get coordinates out of the object,
and then create a new object with those transformed coordinates.

Classes which compose over transformable objects can be adapted with the
`AttrAdapter` class.
See the `SimpleAdapter` or `FnAdapter` for wrapping simple adapting functions.
Implement your own adapter by inheriting from `BaseAdapter`.

See `pd.DataFrameAdapter` for an example of creating an adapter for an external type.

"""
from .base import AttrAdapter, BaseAdapter, FnAdapter, SimpleAdapter, NullAdapter

__all__ = ["BaseAdapter", "SimpleAdapter", "NullAdapter", "FnAdapter", "AttrAdapter"]
