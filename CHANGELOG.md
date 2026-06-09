# Changelog

## Unreleased

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
