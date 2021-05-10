# transformnd

A library providing an API for coordinate transformations,
as well as some common transforms.
The goal is to allow downstream applications which require such transformations
(e.g. image registration) to be generic over anything inheriting from `transformnd.Transform`.

Heavily inspired by/ cribbed directly from
[Philipp Schlegel's work in navis](https://github.com/schlegelp/navis/tree/master/navis/transforms).

## Conventions

`N` coordinates in `D` dimensions are given as a numpy array of shape `(N, D)`.
`transformnd.flatten()` converts arrays into a compatible shape,
and provides a routine for returning to the original shape.

If your `Transform` *subclass* is restricted to certain dimensionalities,
use the `transformnd.limit_ndim()` function in the constructor.
If an *instance* is restricted, set the `ndim` member variable
and call `self._check_ndim()` in `__call__`.

## Additional transforms

Contributions of additional transforms are welcome!
Even if they're only thin wrappers around an external library,
the downstream ecosystem benefits from a consistent API.

Such external libraries should be specified as "extras",
and be contained in a submodule so that they are not immediately imported
with `transformnd`.

Alternatively, consider adopting `transformnd`'s base classes in your own library,
and have your transformation instantly compatible for downstream users.
