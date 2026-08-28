# Changelog

## Unreleased

### Added

- `TransformGraph.relabel_spaces(mapping)` allows relabelling of spaces
- `TransformGraph.ndim(space_ref)` gets the dimensionality of a single space, if it exists
- `TransformGraph.space_ndims()` iterates though all spaces and their dimensionalities
- `GridInterpolation` transform to apply a 1D function to each dimension of the array
  - Intended for use with e.g. `numpy.interp` or 1D `scipy.interpolate` routines,
    and to bring transforms to xarray-like coordinate arrays.
  - This was already possible with `ByDimension` plus some boilerplate

### Changed

- BREAKING: `TransformGraph.add_transform` now takes a `Spaced` (which wraps a transform and its spaces)
- BREAKING: `TransformGraph.__iter__` now yields a tuple of the edge's `Spaced` and the edge data

### Removed

- BREAKING: Transforms no longer have their own `spaces`; this concept only exist on the graph

## 0.7.2 - 2026-06-25

## 0.7.1 - 2026-06-25

## 0.7.0 - 2026-06-25

### Added

- Image transformation tutorial
- Examples are now uploaded with docs

### Changed

- BREAKING: ProjectAxis is now more intuitive to instantiate

## 0.6.0 - 2026-06-18

### Fixed

- Rectangular affines now have the correct input and output dimensions

### Added

- ProjectAxis transformation for adding and dropping axes

### Changed

- Expose all transforms under `.transforms`, even if they are optional
- BREAKING: Rename `GeometryAdapter` to `ShapelyAdapter`

## 0.5.0 - 2026-06-17

### Added

- allow `TransformGraph` to have parallel edges, with arbitrary data for weights
- `TransformGraph.get_sequence(... , *, weight)` argument for path selection
- `TransformSequence.flatten()` method for flattening nested sequences
- `TransformSequence.split()` method for splitting sequences with known intermediate spaces

### Removed

- remove `transforms` argument of `TransformGraph` constructor
- remove `and_inverse` argument of `TransformGraph.add_transform` (must be added explicitly)
- sequence-splitting behaviour when graphs are added (must be split explicitly)
- remove `TransformGraph.add_transforms` method

## 0.4.2 - 2026-06-10

### Fixed

- Affine up/downprojection

## 0.4.1 - 2026-06-09

### Added

- Extras covering `transforms`, `adapters`, and `all`

### Fixed

- Move dask dependency into an extra

## 0.4.0 - 2026-06-09

### Added

- Vector fields transforms `Displacements` and `Coordinates`

## 0.3.0 - 2026-06-04

### Added

- `TransformGraph.add_transform` method

### Fixed

- Handle `Bijection`s as an explicit pair of edges in `TransformGraph`
- Source/target spaces in affine matmul

## 0.2.1 - 2026-06-03

### Fixed

- Update tutorial
- Improve docs around matmul order

## 0.2.0 - 2026-06-02

### Fixed

- Fix space-checking for matmul

### Changed

- Replace `tuple[SpaceRef | None]` with `Spaces(NamedTuple)`
- Require fixed dimensionality for all transforms

### Added

- `Transform`s can now change coordinates' dimensionality
- `TransformSequence` can now be converted to affines in some situations
- More unit tests

## 0.1.0 - 2026-05-26

### Added

- Initial release
