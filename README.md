# transformnd

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GitHub](https://img.shields.io/github/license/clbarnes/transformnd)](https://github.com/clbarnes/transformnd/blob/main/LICENSE)
[![Test Status](https://img.shields.io/github/workflow/status/clbarnes/transformnd/ci)](https://github.com/clbarnes/transformnd/actions/workflows/ci.yaml)
[![Docs Status](https://img.shields.io/github/workflow/status/clbarnes/transformnd/docs?label=docs)](https://clbarnes.github.io/transformnd/)

A library providing an API for coordinate transformations,
as well as some common transforms.
The goal is to allow downstream applications which require such transformations
(e.g. image registration) to be generic over anything inheriting from `transformnd.Transform`.

Heavily inspired by/ cribbed directly from
[Philipp Schlegel's work in navis](https://github.com/schlegelp/navis/tree/master/navis/transforms);
co-developed with [xform](https://github.com/schlegelp/xform/) as a red team prototype.

`N` coordinates in `D` dimensions are given as a numpy array of shape `(N, D)`.
`transformnd.flatten()` converts arrays into a compatible shape,
and provides a routine for returning to the original shape.

`Transform` subclasses which are restricted to certain dimensionalities
can specify this in their `ndim` class variable.
Instances of `Transform` subclasses can further restrict their `ndim`.
Use `self._validate_coords(coords)` in `__call__` to ensure the coordinates
are of valid type and dimensions.

See the [tutorial here](https://github.com/clbarnes/transformnd/blob/main/examples/tutorial.ipynb).

## Additional transforms

Contributions of additional transforms are welcome!
Even if they're only thin wrappers around an external library,
the downstream ecosystem benefits from a consistent API.

Such external libraries should be specified as "extras",
and be contained in a submodule so that they are not immediately imported
with `transformnd`.

Alternatively, consider adopting `transformnd`'s base classes in your own library,
and have your transformation instantly compatible for downstream users.
