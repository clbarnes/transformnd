# Changelog

## Unreleased

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
