# transformnd

A library providing an API for coordinate transformations,
as well as some common transforms.
The goal is to allow downstream applications which require such transformations
(e.g. image registration) to be generic over anything inheriting from `transformnd.Transform`.

Heavily inspired by/ cribbed directly from
[Philipp Schlegel's work in navis](https://github.com/schlegelp/navis/tree/master/navis/transforms).

## Conventions

This library uses the convention that coordinates' dimensions are in the first axis of an array.
That is, an array of `N` points in `D` dimensions will be `DxN`.
This is convenient for:

- matrix multiplication (e.g. for affine transformations)
- `scipy.ndimage.map_coordinates` (e.g. for image registration)
- `numpy.meshgrid` and other functions which return a tuple of 1D coordinate arrays

`NxD` is another common convention; just use `my_array.T` or `numpy.moveaxis(my_array, -1, 0)`.

Additionally, implementors should allow any number of dimensions in the coordinate array;
i.e. arrays of shape `DxIxJxKx...`.
Use the `transformnd.flatten()` function to convert it into `DxN`
and provide a function for the reverse reshaping.

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
