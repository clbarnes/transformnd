# transformnd

[![PyPI - Version](https://img.shields.io/pypi/v/transformnd)](https://pypi.org/project/transformnd/)
[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/clbarnes/transformnd/ci.yaml)](https://github.com/clbarnes/transformnd/actions)
[![Read the Docs](https://img.shields.io/readthedocs/transformnd)](https://transformnd.readthedocs.io)
![PyPI - License](https://img.shields.io/pypi/l/transformnd)

A library providing an API for coordinate transformations,
as well as some common transforms.
The goal is to allow downstream applications which require such transformations
(e.g. image registration) to be generic over anything inheriting from `transformnd.Transform`.

The base classes and utilities are very lightweight with few dependencies, for use as an API; additional transforms and features use extras.

Heavily inspired by/ cribbed directly from
[Philipp Schlegel's work in navis](https://github.com/schlegelp/navis/tree/master/navis/transforms);
co-developed with [xform](https://github.com/schlegelp/xform/) as a red team prototype.

`N` coordinates in `D` dimensions are given as a numpy array of shape `(N, D)`.

`Transform` subclasses which are restricted to certain dimensionalities
can specify this in their `ndim` class variable.
Instances of `Transform` subclasses can further restrict their `ndim`.
Use `self._validate_coords(coords)` in the `apply` method to ensure the coordinates
are of valid type and dimensions.

Additionally, `transformnd` provides an interface for transforming types other than NxD numpy arrays,
and implements these adapters for a few common types.

See the [tutorial here](https://github.com/clbarnes/transformnd/blob/main/examples/tutorial.py).
It is a [marimo](https://marimo.io) notebook.
Open it with `uv run --group tutorial marimo edit examples/tutorial.py`.

## Implemented transforms

- Identity (`transformnd.transforms.Identity`)
- Translation (`transformnd.transforms.Translate`)
- Scale (`transformnd.transforms.Scale`)
- Reflection (`transformnd.transforms.Reflect`)
- Affine (`transformnd.transforms.Affine`)
  - Can be composed efficiently with `@` operator; the right hand operand is effectively applied first
- MapAxis (`transformnd.transforms.MapAxis`): permute coordinate axes
- ByDimension (`transformnd.transforms.ByDimension`): apply transformations to subsets of coordinate axes
- Moving Least Squares, affine (`transformnd.transforms.moving_least_squares.MovingLeastSquares`)
  - uses `movingleastsquares` extra
- Thin Plate Splines (`transformnd.transforms.thinplate.ThinPlateSplines`)
  - uses `thinplatesplines` extra

Arbitrary transforms can be composed into a `TransformSequence` with `transform1 | transform2`.
A graph of transforms between defined spaces can be traversed using the `TransformGraph`.

## Implemented adapters

- Numpy arrays of shape `(..., D, ...)` (`transformnd.adapters.ReshapeAdapter`)
- `meshio.Mesh` (`transformnd.adapters.meshio.MeshAdapter`)
- `pandas.DataFrame` (`transformnd.adapters.pandas.PandasAdapter`)
  - Takes a subset of columns as a coordinate array
- `polars.DataFrame` (`transformnd.adapters.polars.PolarsAdapter`)
  - Similar to the pandas adapter
  - Currently, only scalar columns are supported (e.g. not a single struct column with fields `x`, `y`, `z`)
- Geometries from `shapely` (`transformnd.adapters.shapely.GeometryAdapter`)
- Objects composed of transformable attributes (`transformnd.adapters.AttrAdapter`).

## Additional transforms and adapters

Contributions of additional transforms and adapters are welcome!
Even if they're only thin wrappers around an external library,
the downstream ecosystem benefits from a consistent API.

Such external transformation libraries should be specified as "extras" (`pyproject.toml:project.optional-dependencies`),
and be contained in a submodule so that they are not immediately imported
with `transformnd`.

Alternatively, consider adopting `transformnd`'s base classes in your own library,
and have your transformation instantly compatible for downstream users.

Methods which MUST be implemented:

- `__init__`: should validate parameters and must call the `super()` constructor
- `apply`: should call `_validate_coords` method early to check that the given coordinates are the correct shape

Methods which SHOULD be implemented if applicable:

- `to_device`: if any of the transformation's parameters need to be placed on a specific device (e.g. affine matrices on the GPU)
- `is_identity`: if you can cheaply check whether your transformation is an identity transformation. The base class implementation returns `False`.
- `to_affine`: if your transformation can be represented as an affine matrix. The base class implementation returns `None`.
- `invert`: if your transformation can be inverted (default None if not)
  - This automatically implements `__invert__` (the `~my_transform` operator), which returns `NotImplemented` (probably raising `NotImplementedError`) if `invert` would return `None`.

## Contributing

- Use [`uv`](https://docs.astral.sh/uv/) for environment and dependency management.
  - `uv sync` to set up the environment.
- Use [`prek`](https://prek.j178.dev/) for running pre-commit hooks.
  - `prek install-hooks && prek run --all-files` to get started.
- Use [`just`](https://github.com/casey/just) for common development tasks (format, lint, test, generate docs, run benchmarks).
  - `just` to list commands.
- Docs are generated with [`pdoc`](https://pdoc.dev/) (use `just doc`) and hosted on ReadTheDocs
- `just bump` bumps the version, commits, and tags (but does not push); depends on [`schpet/changelog`](https://github.com/schpet/changelog)

## Thanks

Thanks to contributors

- [Francesca Drummer](https://github.com/FrancescaDr)
- [Lorenzo Cerrone](https://github.com/lorenzocerrone)
- [Maks Hess](https://github.com/MaksHess)
- [Silvia Maria Macrì](https://github.com/SilviaMariaMacri)
